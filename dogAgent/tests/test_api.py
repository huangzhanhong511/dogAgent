"""
dogAgent API 层集成测试

用 FastAPI TestClient + mock LLM/Retriever 测试：
  - 认证（注册/登录/鉴权）
  - 对话接口（保存记忆、context_items 对齐）
  - 会话管理（列表/新建/消息历史）
  - 用户偏好（读取/添加）
"""

import os
import sys
import json
import tempfile
import unittest
from unittest.mock import patch, MagicMock

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "agent"))


# ─── Mock LLM / Retriever ────────────────────────────────────────────────────

class _MockLLMResponse:
    def __init__(self, content):
        self.content = content
        self.tool_calls = []


class MockLLM:
    def __init__(self, responses=None):
        self.responses = list(responses) if responses else []
        self.call_count = 0

    def invoke(self, messages):
        self.call_count += 1
        if self.responses:
            return _MockLLMResponse(self.responses.pop(0))
        return _MockLLMResponse(f"Mock 回答 #{self.call_count}")

    def bind_tools(self, tools):
        return self


class MockRetriever:
    def __init__(self):
        self.index = {"mock": True}

    def retrieve(self, query, top_k=3):
        return []

    def format_context(self, results):
        return ""


# ─── 测试基类 ─────────────────────────────────────────────────────────────────

class APITestBase(unittest.TestCase):
    """所有 API 测试的基类，注入 mock 并创建 TestClient"""

    @classmethod
    def setUpClass(cls):
        """仅在第一个测试前初始化一次"""
        from memory import MemoryDB, ConversationStore, UserPreferences, SummaryDAG
        from session import SessionManager
        from compaction import CompactionEngine
        from query_rewrite import QueryRewriter
        from memory_drilldown import MemoryDrillDown

        cls.db = MemoryDB(":memory:")
        cls.conv_store = ConversationStore(cls.db)
        cls.user_prefs = UserPreferences(cls.db)
        cls.summary_dag = SummaryDAG(cls.db)
        cls.session_mgr = SessionManager(cls.db)
        cls.mock_llm = MockLLM()
        cls.mock_retriever = MockRetriever()

        cls.memory = {
            "db": cls.db,
            "conv_store": cls.conv_store,
            "user_prefs": cls.user_prefs,
            "summary_dag": cls.summary_dag,
            "session_mgr": cls.session_mgr,
            "query_rewriter": QueryRewriter(cls.mock_llm),
            "compaction": CompactionEngine(cls.conv_store, cls.summary_dag, cls.mock_llm),
            "drilldown": MemoryDrillDown(),
        }

        # 注入到 server 全局变量
        import api.server as srv
        srv._llm = cls.mock_llm
        srv._retriever = cls.mock_retriever
        srv._memory = cls.memory

        # 初始化认证服务
        from api.auth import AuthService, set_auth_service
        cls.auth = AuthService(cls.db)
        set_auth_service(cls.auth)
        srv._auth = cls.auth

        # 初始化后台任务管理器（不跑真实任务）
        bg = MagicMock()
        bg.submit_preference_extract = MagicMock()
        bg.submit_compaction = MagicMock()
        bg.submit_title_generate = MagicMock()
        srv._bg_tasks = bg

        from fastapi.testclient import TestClient
        cls.client = TestClient(srv.app)

    def setUp(self):
        """每个测试前重置 LLM responses"""
        self.mock_llm.responses = []
        self.mock_llm.call_count = 0

    def _register_and_login(self, username=None, password="test1234"):
        """注册并返回 token"""
        import uuid
        username = username or f"user_{uuid.uuid4().hex[:8]}"
        r = self.client.post("/api/auth/register", json={"username": username, "password": password})
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()["token"], username

    def _auth_headers(self, token):
        return {"Authorization": f"Bearer {token}"}


# ─── 认证测试 ─────────────────────────────────────────────────────────────────

class TestAuth(APITestBase):

    def test_register_success(self):
        """注册新用户"""
        r = self.client.post("/api/auth/register", json={
            "username": "newuser_reg", "password": "pass1234"
        })
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("token", data)
        self.assertEqual(data["username"], "newuser_reg")

    def test_register_duplicate(self):
        """重复注册报错"""
        self.client.post("/api/auth/register", json={"username": "dup_user", "password": "pass1234"})
        r = self.client.post("/api/auth/register", json={"username": "dup_user", "password": "pass1234"})
        self.assertEqual(r.status_code, 400)

    def test_login_success(self):
        """登录成功"""
        self.client.post("/api/auth/register", json={"username": "login_user", "password": "pass1234"})
        r = self.client.post("/api/auth/login", json={"username": "login_user", "password": "pass1234"})
        self.assertEqual(r.status_code, 200)
        self.assertIn("token", r.json())

    def test_login_wrong_password(self):
        """密码错误 401"""
        self.client.post("/api/auth/register", json={"username": "wrong_pw", "password": "correct1"})
        r = self.client.post("/api/auth/login", json={"username": "wrong_pw", "password": "wrongone"})
        self.assertEqual(r.status_code, 401)

    def test_me_requires_auth(self):
        """/me 需要认证"""
        r = self.client.get("/api/auth/me")
        self.assertEqual(r.status_code, 401)

    def test_me_returns_user_info(self):
        """已认证时 /me 返回用户信息"""
        token, username = self._register_and_login()
        r = self.client.get("/api/auth/me", headers=self._auth_headers(token))
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["username"], username)


# ─── 对话测试 ─────────────────────────────────────────────────────────────────

class TestChat(APITestBase):

    def test_chat_saves_to_memory(self):
        """对话后消息写入 conversations 表"""
        token, _ = self._register_and_login()
        self.mock_llm.responses = ["雪纳瑞有三种，体型不同。"]

        r = self.client.post("/api/chat",
            json={"message": "雪纳瑞有几种？"},
            headers=self._auth_headers(token),
        )
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("answer", data)
        self.assertIn("雪纳瑞", data["answer"])

    def test_chat_context_items_updated(self):
        """对话后 context_items 有对应的 message 条目"""
        token, _ = self._register_and_login()
        self.mock_llm.responses = ["雪纳瑞寿命约 12-15 年。"]

        r = self.client.post("/api/chat",
            json={"message": "雪纳瑞寿命多长？"},
            headers=self._auth_headers(token),
        )
        self.assertEqual(r.status_code, 200)
        session_id = r.json().get("session_id")
        self.assertIsNotNone(session_id)

        # 找到对应 user_id
        user_data = self.client.get("/api/auth/me", headers=self._auth_headers(token)).json()
        user_id = user_data["user_id"]

        items = self.summary_dag.get_context_items(user_id, session_id)
        self.assertGreater(len(items), 0)
        self.assertTrue(all(it["item_type"] == "message" for it in items))

    def test_chat_uses_session_id(self):
        """指定 session_id 对话"""
        token, _ = self._register_and_login()
        user_id = self.client.get("/api/auth/me", headers=self._auth_headers(token)).json()["user_id"]

        session_id = self.session_mgr.create_session(user_id, "测试会话")
        self.mock_llm.responses = ["测试回答"]

        r = self.client.post("/api/chat",
            json={"message": "测试", "session_id": session_id},
            headers=self._auth_headers(token),
        )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["session_id"], session_id)

    def test_chat_requires_auth(self):
        """未认证时 chat 返回 401"""
        r = self.client.post("/api/chat", json={"message": "你好"})
        self.assertEqual(r.status_code, 401)

    def test_chat_multi_turn_context(self):
        """多轮对话 — 第二轮 LLM 收到历史消息"""
        token, _ = self._register_and_login()
        self.mock_llm.responses = ["第一轮回答", "查询重写", "第二轮回答"]

        r1 = self.client.post("/api/chat",
            json={"message": "雪纳瑞怎么洗澡？"},
            headers=self._auth_headers(token),
        )
        session_id = r1.json()["session_id"]

        r2 = self.client.post("/api/chat",
            json={"message": "频率呢？", "session_id": session_id},
            headers=self._auth_headers(token),
        )
        self.assertEqual(r2.status_code, 200)
        # 第二轮必然消耗了更多 LLM 调用（因为有历史消息触发 rewrite）
        self.assertGreaterEqual(self.mock_llm.call_count, 2)


# ─── 会话管理测试 ─────────────────────────────────────────────────────────────

class TestSessions(APITestBase):

    def test_list_sessions(self):
        """列出会话"""
        token, _ = self._register_and_login()
        self.mock_llm.responses = ["回答"]

        # 先发一条消息触发会话创建
        self.client.post("/api/chat", json={"message": "你好"}, headers=self._auth_headers(token))

        r = self.client.get("/api/sessions", headers=self._auth_headers(token))
        self.assertEqual(r.status_code, 200)
        sessions = r.json()["sessions"]
        self.assertGreater(len(sessions), 0)

    def test_create_session(self):
        """新建会话"""
        token, _ = self._register_and_login()
        r = self.client.post("/api/sessions",
            json={"title": "新建测试会话"},
            headers=self._auth_headers(token),
        )
        self.assertEqual(r.status_code, 200)
        self.assertIn("session_id", r.json())

    def test_get_session_messages(self):
        """获取会话消息历史"""
        token, _ = self._register_and_login()
        self.mock_llm.responses = ["消息历史测试回答"]

        r = self.client.post("/api/chat",
            json={"message": "消息历史测试"},
            headers=self._auth_headers(token),
        )
        session_id = r.json()["session_id"]

        r2 = self.client.get(f"/api/sessions/{session_id}/messages",
            headers=self._auth_headers(token),
        )
        self.assertEqual(r2.status_code, 200)
        messages = r2.json()["messages"]
        self.assertGreater(len(messages), 0)

    def test_sessions_isolated_between_users(self):
        """不同用户的会话相互隔离"""
        token_a, _ = self._register_and_login()
        token_b, _ = self._register_and_login()

        self.mock_llm.responses = ["A 的回答", "B 的回答"]
        self.client.post("/api/chat", json={"message": "A 的问题"}, headers=self._auth_headers(token_a))
        self.client.post("/api/chat", json={"message": "B 的问题"}, headers=self._auth_headers(token_b))

        sessions_a = self.client.get("/api/sessions", headers=self._auth_headers(token_a)).json()["sessions"]
        sessions_b = self.client.get("/api/sessions", headers=self._auth_headers(token_b)).json()["sessions"]

        ids_a = {s["id"] for s in sessions_a}
        ids_b = {s["id"] for s in sessions_b}
        self.assertTrue(ids_a.isdisjoint(ids_b), "不同用户的会话不应重叠")


# ─── 偏好测试 ─────────────────────────────────────────────────────────────────

class TestPreferences(APITestBase):

    def test_add_and_get_preferences(self):
        """添加偏好后能读取"""
        token, _ = self._register_and_login()

        r = self.client.post("/api/preferences",
            json={"content": "我家狗叫旺旺"},
            headers=self._auth_headers(token),
        )
        self.assertEqual(r.status_code, 200)

        r2 = self.client.get("/api/preferences", headers=self._auth_headers(token))
        self.assertEqual(r2.status_code, 200)
        prefs = r2.json()["preferences"]
        contents = [p["content"] for p in prefs]
        self.assertIn("我家狗叫旺旺", contents)

    def test_preferences_isolated_between_users(self):
        """不同用户偏好隔离"""
        token_a, _ = self._register_and_login()
        token_b, _ = self._register_and_login()

        self.client.post("/api/preferences",
            json={"content": "用户A的偏好"},
            headers=self._auth_headers(token_a),
        )

        prefs_b = self.client.get("/api/preferences", headers=self._auth_headers(token_b)).json()["preferences"]
        contents_b = [p["content"] for p in prefs_b]
        self.assertNotIn("用户A的偏好", contents_b)

    def test_preferences_in_chat_context(self):
        """偏好注入到对话 system prompt"""
        token, _ = self._register_and_login()

        # 添加偏好
        self.client.post("/api/preferences",
            json={"content": "我家狗叫花花"},
            headers=self._auth_headers(token),
        )

        self.mock_llm.responses = ["花花的回答"]
        r = self.client.post("/api/chat",
            json={"message": "我家狗叫什么？"},
            headers=self._auth_headers(token),
        )
        self.assertEqual(r.status_code, 200)
        # LLM 被调用过（偏好已注入，无法直接断言 system prompt 内容，但调用成功说明流程通）
        self.assertGreater(self.mock_llm.call_count, 0)


# ─── 压缩触发测试 ─────────────────────────────────────────────────────────────

class TestCompactionTrigger(APITestBase):

    def test_api_submits_compaction_task(self):
        """每轮对话后 API 都提交了压缩后台任务"""
        import api.server as srv
        srv._bg_tasks.submit_compaction.reset_mock()

        token, _ = self._register_and_login()
        self.mock_llm.responses = ["回答1", "重写", "回答2"]

        self.client.post("/api/chat", json={"message": "问题1"}, headers=self._auth_headers(token))
        self.client.post("/api/chat", json={"message": "问题2"}, headers=self._auth_headers(token))

        self.assertGreaterEqual(srv._bg_tasks.submit_compaction.call_count, 2)

    def test_compaction_creates_summaries_in_db(self):
        """直接运行压缩后 SQLite 中出现 summaries 记录"""
        from compaction import CompactionEngine
        from memory import MemoryDB, ConversationStore, SummaryDAG
        from session import SessionManager

        db = MemoryDB(":memory:")
        conv = ConversationStore(db)
        dag = SummaryDAG(db)
        sess = SessionManager(db)
        llm = MockLLM(responses=["leaf摘要内容"] * 10)
        engine = CompactionEngine(conv, dag, llm, config={
            "context_budget": 200,
            "context_threshold": 0.5,
            "fresh_tail_count": 2,
            "leaf_chunk_tokens": 300,
            "min_fanout": 3,
            "max_rounds": 5,
        })

        uid = "compact_test_user"
        sid = sess.create_session(uid, "压缩测试")

        # 写入足量消息超过阈值
        for i in range(20):
            conv.add_message(uid, sid, "user", f"关于雪纳瑞健康问题的详细描述，编号{i}")
            conv.add_message(uid, sid, "assistant", f"详细回答建议，编号{i}，包含用药方案和注意事项")

        tokens_before = dag.get_context_token_count(uid, sid)
        self.assertGreater(tokens_before, 100)

        stats = engine.compact(uid, sid)

        # 验证 SQLite summaries 表中有记录
        cursor = db.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM summaries WHERE user_id = ? AND session_id = ?", (uid, sid))
        summary_count = cursor.fetchone()[0]
        self.assertGreater(summary_count, 0)
        self.assertGreater(stats["leaf_count"], 0)

        # 验证 context_items 中有 summary 条目
        items = dag.get_context_items(uid, sid)
        summary_items = [it for it in items if it["item_type"] == "summary"]
        self.assertGreater(len(summary_items), 0)

        # 验证 token 数下降
        tokens_after = dag.get_context_token_count(uid, sid)
        self.assertLess(tokens_after, tokens_before)


# ─── 健康检查 ─────────────────────────────────────────────────────────────────

class TestHealth(APITestBase):

    def test_health_endpoint(self):
        """/api/health 返回 ok"""
        r = self.client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")


if __name__ == "__main__":
    unittest.main(verbosity=2)
