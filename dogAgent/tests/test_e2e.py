"""
dogAgent 端到端集成测试

用 Mock LLM 验证整个对话链路：
  检索 → 记忆 → 偏好 → 查询重写 → DAG 压缩 → 会话管理

无需真实 API key，全部在本地完成。
"""

import os
import sys
import json
import tempfile
import unittest
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

# 设置项目路径
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "agent"))


# ─── Mock LLM ────────────────────────────────────────────────────────────────

class MockLLMResponse:
    """模拟 LLM 返回"""
    def __init__(self, content: str):
        self.content = content


class MockLLM:
    """
    可编程的 Mock LLM。
    可以设置一系列预定义回复，或用默认回复。
    """
    def __init__(self, responses=None):
        self.responses = list(responses) if responses else []
        self.call_count = 0
        self.call_history = []

    def invoke(self, messages):
        """模拟 LLM 调用"""
        self.call_count += 1
        # 记录调用
        msg_summary = []
        for m in messages:
            role = type(m).__name__
            content = m.content[:100] if hasattr(m, 'content') else str(m)[:100]
            msg_summary.append(f"{role}: {content}")
        self.call_history.append(msg_summary)

        # 返回预定义回复或默认回复
        if self.responses:
            resp = self.responses.pop(0)
            return MockLLMResponse(resp)
        return MockLLMResponse(f"Mock 回答 #{self.call_count}: 雪纳瑞是一种很聪明的狗。")


# ─── Mock Retriever ───────────────────────────────────────────────────────────

@dataclass
class MockRetrievalResult:
    title: str = "迷你雪纳瑞概述"
    path: str = "wiki/01-品种百科/迷你雪纳瑞概述.md"
    score: float = 0.85
    match_reasons: list = field(default_factory=lambda: ["关键词匹配(0.90)"])
    content: str = "迷你雪纳瑞是一种小型犬，体重约 5-8 kg，性格活泼聪明。"
    sections_matched: list = field(default_factory=list)


class MockRetriever:
    """模拟 Wiki 检索器"""
    def __init__(self, results=None):
        self.index = {"mock": True}  # 非空，表示索引已加载
        self._results = results or [MockRetrievalResult()]
        self.retrieve_calls = []

    def retrieve(self, query, top_k=3):
        self.retrieve_calls.append(query)
        return self._results[:top_k]

    def format_context(self, results):
        if not results:
            return ""
        parts = []
        for r in results:
            parts.append(f"# {r.title}\n\n{r.content}")
        return "\n\n---\n\n".join(parts)


# ─── 测试基类 ─────────────────────────────────────────────────────────────────

class E2ETestBase(unittest.TestCase):
    """为所有 E2E 测试提供共用的 setup/teardown"""

    def setUp(self):
        """创建临时目录和内存数据库"""
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = ":memory:"

        # 导入所有模块
        from memory import MemoryDB, ConversationStore, UserPreferences, SummaryDAG
        from session import SessionManager
        from compaction import CompactionEngine
        from query_rewrite import QueryRewriter

        # 创建数据库和组件
        self.db = MemoryDB(self.db_path)
        self.conv_store = ConversationStore(self.db)
        self.user_prefs = UserPreferences(self.db)
        self.summary_dag = SummaryDAG(self.db)
        self.session_mgr = SessionManager(self.db)

        # Mock LLM
        self.llm = MockLLM()
        self.retriever = MockRetriever()

        # CompactionEngine 和 QueryRewriter
        self.compaction = CompactionEngine(self.conv_store, self.summary_dag, self.llm)
        self.query_rewriter = QueryRewriter(self.llm)

        # 组装 memory dict（和 chat.py 中一致）
        self.memory = {
            "db": self.db,
            "conv_store": self.conv_store,
            "user_prefs": self.user_prefs,
            "summary_dag": self.summary_dag,
            "session_mgr": self.session_mgr,
            "query_rewriter": self.query_rewriter,
            "compaction": self.compaction,
        }

        self.user_id = "test_user"

    def tearDown(self):
        """清理"""
        try:
            self.db.conn.close()
        except:
            pass

    def simulate_turn(self, user_input: str, session_id: str) -> str:
        """
        模拟一轮完整对话（和 chat.py 的主循环逻辑一致）。
        返回 assistant 的回答。
        """
        from chat import build_memory_context, build_messages, SYSTEM_PROMPT

        memory = self.memory

        # 1. 查询重写
        recent_msgs = memory["conv_store"].get_recent(self.user_id, session_id, limit=10)
        rewritten = memory["query_rewriter"].rewrite(user_input, recent_msgs)

        # 2. Wiki 检索
        results = self.retriever.retrieve(rewritten, top_k=3)
        wiki_context = self.retriever.format_context(results)

        # 3. 记忆上下文
        memory_context = build_memory_context(memory, self.user_id, session_id)

        # 4. 组装消息 + 调用 LLM（从 context_items 取 fresh tail）
        tail_msgs = memory["summary_dag"].get_context_messages(self.user_id, session_id)
        messages = build_messages(SYSTEM_PROMPT, memory_context, wiki_context, tail_msgs, user_input)
        response = self.llm.invoke(messages)
        answer = response.content

        # 5. 保存
        memory["conv_store"].add_message(self.user_id, session_id, "user", user_input)
        memory["conv_store"].add_message(self.user_id, session_id, "assistant", answer)
        memory["session_mgr"].touch_session(session_id)

        # 6. 压缩
        if memory["compaction"]:
            memory["compaction"].check_and_compact(self.user_id, session_id)

        return answer


# ─── 测试用例 ─────────────────────────────────────────────────────────────────

class TestBasicConversation(E2ETestBase):
    """测试基本对话流程"""

    def test_single_turn(self):
        """单轮对话：问 → 检索 → 回答 → 保存"""
        session_id = self.session_mgr.create_session(self.user_id)

        answer = self.simulate_turn("雪纳瑞有几种？", session_id)

        # 验证有回答
        self.assertTrue(len(answer) > 0)

        # 验证消息已保存
        msgs = self.conv_store.get_recent(self.user_id, session_id, limit=10)
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["role"], "user")
        self.assertEqual(msgs[0]["content"], "雪纳瑞有几种？")
        self.assertEqual(msgs[1]["role"], "assistant")

        # 验证检索器被调用
        self.assertEqual(len(self.retriever.retrieve_calls), 1)

    def test_multi_turn(self):
        """多轮对话：验证历史消息正确传递"""
        session_id = self.session_mgr.create_session(self.user_id)

        # 设置多轮回复
        # QueryRewriter 在有历史消息时也会调用 LLM（消耗一个 response）
        # 第 1 轮: 无历史，不重写 → 只消耗 1 个（对话）
        # 第 2 轮: 有历史，重写消耗 1 个 + 对话消耗 1 个 = 2
        # 第 3 轮: 有历史，重写消耗 1 个 + 对话消耗 1 个 = 2
        self.llm.responses = [
            "雪纳瑞有三种：迷你型、标准型和巨型。",        # turn 1: 对话
            "雪纳瑞体重问题",                               # turn 2: rewrite
            "迷你雪纳瑞体重约 5-8 kg，标准型约 14-20 kg。", # turn 2: 对话
            "雪纳瑞寿命问题",                               # turn 3: rewrite
            "雪纳瑞寿命一般 12-15 年。",                    # turn 3: 对话
        ]

        a1 = self.simulate_turn("雪纳瑞有几种？", session_id)
        a2 = self.simulate_turn("它们体重分别是多少？", session_id)
        a3 = self.simulate_turn("寿命呢？", session_id)

        # 验证 3 轮对话保存了 6 条消息
        msgs = self.conv_store.get_recent(self.user_id, session_id, limit=20)
        self.assertEqual(len(msgs), 6)

        # 验证 LLM 被调用 3 次（对话） + 可能的 query rewrite 调用
        self.assertGreaterEqual(self.llm.call_count, 3)

        # 验证回答内容
        self.assertIn("三种", a1)
        self.assertIn("kg", a2)
        self.assertIn("寿命", a3)


class TestMemoryContext(E2ETestBase):
    """测试记忆上下文构建"""

    def test_memory_context_with_prefs(self):
        """用户偏好注入到记忆上下文"""
        from chat import build_memory_context

        session_id = self.session_mgr.create_session(self.user_id)

        # 添加偏好
        self.user_prefs.add_preference(self.user_id, "我家狗叫旺旺")
        self.user_prefs.add_preference(self.user_id, "旺旺是迷你雪纳瑞，3岁")

        ctx = build_memory_context(self.memory, self.user_id, session_id)

        self.assertIn("用户偏好", ctx)
        self.assertIn("旺旺", ctx)
        self.assertIn("迷你雪纳瑞", ctx)

    def test_memory_context_with_summaries(self):
        """DAG 摘要注入到记忆上下文"""
        from chat import build_memory_context

        session_id = self.session_mgr.create_session(self.user_id)

        # 手动插入一个摘要并同步到 context_items
        s_id = self.summary_dag.add_summary(
            self.user_id, session_id,
            depth=0,
            content="用户询问了雪纳瑞的品种分类和体重特征。",
            source_start_id=1,
            source_end_id=4,
        )
        self.summary_dag.append_context_summary(self.user_id, session_id, s_id)

        ctx = build_memory_context(self.memory, self.user_id, session_id)

        self.assertIn("对话摘要", ctx)
        self.assertIn("品种分类", ctx)

    def test_memory_context_empty(self):
        """无记忆时上下文为空"""
        from chat import build_memory_context

        session_id = self.session_mgr.create_session(self.user_id)
        ctx = build_memory_context(self.memory, self.user_id, session_id)

        self.assertEqual(ctx, "")

    def test_build_messages_structure(self):
        """验证消息列表结构正确"""
        from chat import build_messages, SYSTEM_PROMPT
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

        recent = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！我是 dogAgent。"},
        ]

        msgs = build_messages(
            SYSTEM_PROMPT,
            "## 用户偏好\n- 我家狗叫旺旺",
            "# 迷你雪纳瑞\n小型犬...",
            recent,
            "雪纳瑞吃什么好？"
        )

        # SystemMessage 包含偏好
        self.assertIsInstance(msgs[0], SystemMessage)
        self.assertIn("用户偏好", msgs[0].content)
        self.assertIn("旺旺", msgs[0].content)

        # 历史对话
        self.assertIsInstance(msgs[1], HumanMessage)
        self.assertEqual(msgs[1].content, "你好")
        self.assertIsInstance(msgs[2], AIMessage)

        # 最后一条是 wiki 上下文 + 用户问题
        self.assertIsInstance(msgs[3], HumanMessage)
        self.assertIn("迷你雪纳瑞", msgs[3].content)
        self.assertIn("雪纳瑞吃什么好", msgs[3].content)


class TestSessionManagement(E2ETestBase):
    """测试会话管理功能"""

    def test_create_and_switch_sessions(self):
        """创建多个会话并切换"""
        s1 = self.session_mgr.create_session(self.user_id, "关于饮食")
        s2 = self.session_mgr.create_session(self.user_id, "关于健康")

        sessions = self.session_mgr.list_sessions(self.user_id)
        self.assertEqual(len(sessions), 2)

        # 在不同会话中对话
        self.conv_store.add_message(self.user_id, s1, "user", "狗粮怎么选？")
        self.conv_store.add_message(self.user_id, s2, "user", "白内障怎么治？")

        msgs1 = self.conv_store.get_recent(self.user_id, s1, limit=10)
        msgs2 = self.conv_store.get_recent(self.user_id, s2, limit=10)

        self.assertEqual(len(msgs1), 1)
        self.assertIn("狗粮", msgs1[0]["content"])
        self.assertEqual(len(msgs2), 1)
        self.assertIn("白内障", msgs2[0]["content"])

    def test_auto_title_generation(self):
        """自动标题生成触发条件"""
        session_id = self.session_mgr.create_session(self.user_id)

        # 少于 3 轮不触发
        self.conv_store.add_message(self.user_id, session_id, "user", "你好")
        self.conv_store.add_message(self.user_id, session_id, "assistant", "你好！")
        self.assertFalse(self.session_mgr.needs_title(session_id))

        # 加到 3 轮
        for i in range(4):
            self.conv_store.add_message(self.user_id, session_id, "user", f"问题 {i}")
            self.conv_store.add_message(self.user_id, session_id, "assistant", f"回答 {i}")

        self.assertTrue(self.session_mgr.needs_title(session_id))

        # 生成标题后不再触发
        self.session_mgr.update_title(session_id, "雪纳瑞健康问题")
        self.assertFalse(self.session_mgr.needs_title(session_id))

    def test_ensure_session(self):
        """ensure_session 自动创建或复用"""
        # 第一次调用：创建新 session
        s1 = self.session_mgr.ensure_session(self.user_id)
        self.assertIsNotNone(s1)

        # 第二次调用：复用
        s2 = self.session_mgr.ensure_session(self.user_id)
        self.assertEqual(s1, s2)


class TestUserPreferences(E2ETestBase):
    """测试用户偏好流程"""

    def test_add_and_retrieve_prefs(self):
        """添加偏好并检索"""
        self.user_prefs.add_preference(self.user_id, "我家狗叫旺旺")
        self.user_prefs.add_preference(self.user_id, "旺旺 3 岁")
        self.user_prefs.add_preference(self.user_id, "对鸡肉过敏")

        prefs = self.user_prefs.get_active(self.user_id)
        self.assertEqual(len(prefs), 3)
        contents = [p["content"] for p in prefs]
        self.assertIn("我家狗叫旺旺", contents)
        self.assertIn("对鸡肉过敏", contents)

    def test_prefs_text_for_prompt(self):
        """偏好纯文本格式（注入 prompt）"""
        self.user_prefs.add_preference(self.user_id, "迷你雪纳瑞")
        self.user_prefs.add_preference(self.user_id, "5岁公犬")

        text = self.user_prefs.get_active_text(self.user_id)
        self.assertIn("迷你雪纳瑞", text)
        self.assertIn("5岁公犬", text)

    def test_prefs_in_conversation(self):
        """偏好影响对话上下文"""
        from chat import build_memory_context

        session_id = self.session_mgr.create_session(self.user_id)
        self.user_prefs.add_preference(self.user_id, "对鸡肉过敏")

        ctx = build_memory_context(self.memory, self.user_id, session_id)
        self.assertIn("鸡肉过敏", ctx)


class TestQueryRewrite(E2ETestBase):
    """测试查询重写"""

    def test_rewrite_with_context(self):
        """有上下文时查询重写"""
        # 设置 LLM mock 回复重写后的查询
        self.llm.responses = ["雪纳瑞迷你型的体重范围是多少"]

        recent_msgs = [
            {"role": "user", "content": "雪纳瑞有几种？"},
            {"role": "assistant", "content": "雪纳瑞有三种：迷你型、标准型和巨型。"},
        ]

        rewritten = self.query_rewriter.rewrite("迷你型多重？", recent_msgs)

        # 应该调用了 LLM
        self.assertGreaterEqual(self.llm.call_count, 1)
        # 重写后的查询应该更完整
        self.assertTrue(len(rewritten) > 0)

    def test_rewrite_no_context(self):
        """无上下文时不重写"""
        original = "雪纳瑞吃什么？"
        rewritten = self.query_rewriter.rewrite(original, [])

        # 无上下文时应该返回原查询
        self.assertEqual(rewritten, original)


class TestDAGCompaction(E2ETestBase):
    """测试 DAG 摘要压缩"""

    def test_compaction_trigger(self):
        """大量消息后触发压缩"""
        session_id = self.session_mgr.create_session(self.user_id)

        # 预设足够多的 LLM 响应（对话 + 压缩摘要）
        self.llm.responses = []
        for i in range(30):
            self.llm.responses.append(f"回答 {i}")
        # 额外的压缩摘要回复
        self.llm.responses.extend([
            "用户讨论了雪纳瑞的多个健康问题。",
            "用户讨论了雪纳瑞的多个健康问题。",
            "用户讨论了雪纳瑞的多个健康问题。",
        ])

        # 模拟 20 轮对话
        for i in range(20):
            self.simulate_turn(f"关于雪纳瑞的问题 {i}", session_id)

        # 验证消息已保存
        msgs = self.conv_store.get_recent(self.user_id, session_id, limit=100)
        self.assertGreater(len(msgs), 0)

    def test_summary_creation(self):
        """手动创建摘要并检索"""
        session_id = self.session_mgr.create_session(self.user_id)

        # 添加一些消息
        for i in range(6):
            self.conv_store.add_message(self.user_id, session_id, "user", f"问题 {i}")
            self.conv_store.add_message(self.user_id, session_id, "assistant", f"回答 {i}")

        # 创建摘要
        sid = self.summary_dag.add_summary(
            self.user_id, session_id,
            depth=0,
            content="用户询问了 6 个关于雪纳瑞的问题。",
            source_start_id=1,
            source_end_id=12,
        )
        self.assertIsNotNone(sid)

        # 验证摘要已存储
        stored = self.summary_dag.get_summary_by_id(sid)
        self.assertIsNotNone(stored)
        self.assertIn("6 个", stored["content"])


class TestDAGDrillDown(E2ETestBase):
    """测试 DAG 回溯钻取（Lossless Claw）"""

    def test_drilldown_to_messages(self):
        """从摘要钻取到原始消息"""
        session_id = self.session_mgr.create_session(self.user_id)

        # 添加原始消息
        self.conv_store.add_message(self.user_id, session_id, "user", "旺旺耳朵红红的怎么办")
        self.conv_store.add_message(self.user_id, session_id, "assistant", "建议使用耳肤灵滴耳液，每天两次")
        self.conv_store.add_message(self.user_id, session_id, "user", "需要吃消炎药吗")
        self.conv_store.add_message(self.user_id, session_id, "assistant", "可以口服阿莫西林克拉维酸钾，每公斤12.5mg")

        # 创建 leaf 摘要（指向消息 1-4）
        leaf_id = self.summary_dag.add_summary(
            self.user_id, session_id, depth=0,
            content="讨论了旺旺的耳道感染治疗方案",
            source_start_id=1, source_end_id=4,
            child_ids=["msg_range:1-4"],
            child_types=["message_range"],
        )

        # 钻取到原始消息
        children = self.summary_dag.drill_down(leaf_id, self.user_id, self.conv_store)
        self.assertTrue(len(children) > 0)
        self.assertEqual(children[0]["type"], "messages")
        self.assertTrue(len(children[0]["messages"]) > 0)

    def test_drilldown_integration_in_context(self):
        """摘要以 XML 格式注入 context（LLM 通过 memory_expand tool 按需展开）"""
        from chat import build_memory_context

        session_id = self.session_mgr.create_session(self.user_id)

        # 添加消息 + 摘要
        self.conv_store.add_message(self.user_id, session_id, "user", "推荐什么耳药水")
        self.conv_store.add_message(self.user_id, session_id, "assistant", "推荐耳肤灵")

        s_id = self.summary_dag.add_summary(
            self.user_id, session_id, depth=0,
            content="讨论了耳药水推荐",
            source_start_id=1, source_end_id=2,
            child_ids=["msg_range:1-2"],
            child_types=["message_range"],
        )
        self.summary_dag.append_context_summary(self.user_id, session_id, s_id)

        from memory_drilldown import MemoryDrillDown
        self.memory["drilldown"] = MemoryDrillDown()

        ctx = build_memory_context(self.memory, self.user_id, session_id)
        # LLM 驱动展开：context 只包含 XML 摘要，不主动注入钻取结果
        self.assertIn("对话摘要", ctx)
        self.assertIn("耳药水推荐", ctx)
        self.assertNotIn("历史对话详情", ctx)

    def test_no_drilldown_for_normal_query(self):
        """普通查询不触发钻取"""
        from chat import build_memory_context

        session_id = self.session_mgr.create_session(self.user_id)
        s_id = self.summary_dag.add_summary(
            self.user_id, session_id, depth=0,
            content="讨论了饮食问题",
            source_start_id=1, source_end_id=2,
        )
        self.summary_dag.append_context_summary(self.user_id, session_id, s_id)

        from memory_drilldown import MemoryDrillDown
        self.memory["drilldown"] = MemoryDrillDown()

        ctx = build_memory_context(self.memory, self.user_id, session_id)
        # 应该有摘要但没有钻取详情
        self.assertIn("对话摘要", ctx)
        self.assertNotIn("历史对话详情", ctx)


class TestHandleCommand(E2ETestBase):
    """测试命令处理"""

    def test_handle_command_new(self):
        """测试 /new 命令"""
        from chat import _handle_command

        old_session = self.session_mgr.create_session(self.user_id)
        new_session = _handle_command("/new", self.memory, self.user_id, old_session)

        self.assertNotEqual(old_session, new_session)

    def test_handle_command_prefs(self):
        """测试 /prefs 命令不报错"""
        from chat import _handle_command

        session_id = self.session_mgr.create_session(self.user_id)
        self.user_prefs.add_preference(self.user_id, "我家狗叫旺旺")

        # 不应抛异常
        result = _handle_command("/prefs", self.memory, self.user_id, session_id)
        self.assertEqual(result, session_id)

    def test_handle_command_add_pref(self):
        """测试 /pref 内容 命令"""
        from chat import _handle_command

        session_id = self.session_mgr.create_session(self.user_id)
        _handle_command("/pref 我家狗叫旺旺", self.memory, self.user_id, session_id)

        prefs = self.user_prefs.get_active(self.user_id)
        self.assertEqual(len(prefs), 1)
        self.assertEqual(prefs[0]["content"], "我家狗叫旺旺")

    def test_handle_command_sessions(self):
        """测试 /sessions 命令"""
        from chat import _handle_command

        s1 = self.session_mgr.create_session(self.user_id, "会话一")
        result = _handle_command("/sessions", self.memory, self.user_id, s1)
        self.assertEqual(result, s1)

    def test_handle_command_help(self):
        """测试 /help 命令"""
        from chat import _handle_command

        session_id = self.session_mgr.create_session(self.user_id)
        result = _handle_command("/help", self.memory, self.user_id, session_id)
        self.assertEqual(result, session_id)


class TestFullFlow(E2ETestBase):
    """完整流程集成测试"""

    def test_complete_conversation_flow(self):
        """
        模拟完整的用户交互流程：
        1. 创建会话
        2. 设置偏好
        3. 多轮对话
        4. 验证记忆上下文
        5. 新建会话
        6. 跨 session 记忆
        """
        from chat import build_memory_context, _handle_command

        # 1. 创建会话
        session_id = self.session_mgr.ensure_session(self.user_id)
        self.assertIsNotNone(session_id)

        # 2. 设置偏好
        _handle_command("/pref 我家狗叫旺旺，迷你雪纳瑞，3岁", self.memory, self.user_id, session_id)
        prefs = self.user_prefs.get_active(self.user_id)
        self.assertEqual(len(prefs), 1)

        # 3. 多轮对话
        # turn 1: 无历史，不重写 → 1 个 response（对话）
        # turn 2: 有历史，重写消耗 1 个 + 对话消耗 1 个 = 2 个 response
        self.llm.responses = [
            "雪纳瑞需要定期剪毛和清理耳朵。",                        # turn 1: 对话
            "旺旺运动量查询",                                         # turn 2: rewrite 消耗
            "旺旺 3 岁正是活力充沛的年纪，建议每天散步 30-60 分钟。", # turn 2: 对话
        ]

        a1 = self.simulate_turn("雪纳瑞怎么护理？", session_id)
        self.assertIn("剪毛", a1)

        a2 = self.simulate_turn("旺旺多大运动量合适？", session_id)
        self.assertIn("散步", a2)

        # 4. 验证记忆上下文包含偏好
        ctx = build_memory_context(self.memory, self.user_id, session_id)
        self.assertIn("旺旺", ctx)

        # 5. 新建会话
        new_session = _handle_command("/new", self.memory, self.user_id, session_id)
        self.assertNotEqual(new_session, session_id)

        # 新会话消息为空
        new_msgs = self.conv_store.get_recent(self.user_id, new_session, limit=10)
        self.assertEqual(len(new_msgs), 0)

        # 旧会话消息仍在
        old_msgs = self.conv_store.get_recent(self.user_id, session_id, limit=10)
        self.assertEqual(len(old_msgs), 4)

        # 6. 偏好跨 session 保持
        prefs_in_new = self.user_prefs.get_active(self.user_id)
        self.assertEqual(len(prefs_in_new), 1)
        self.assertIn("旺旺", prefs_in_new[0]["content"])


if __name__ == "__main__":
    unittest.main(verbosity=2)