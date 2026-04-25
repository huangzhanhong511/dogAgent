#!/usr/bin/env python3
"""
CompactionEngine 核心算法单元测试

覆盖三个 LCM 关键函数：
  - _resolve_fresh_tail_ordinal
  - _select_oldest_leaf_chunk
  - _select_oldest_chunk_at_depth
以及端到端压缩流程（compact）。
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.memory import MemoryDB, ConversationStore, SummaryDAG
from agent.session import SessionManager
from agent.compaction import CompactionEngine


class MockLLM:
    """固定返回摘要文本的 mock LLM"""
    def invoke(self, prompt: str):
        class R:
            content = "mock摘要内容。\nExpand for details about: 细节A、细节B"
        return R()


def _make_engine(config=None):
    db = MemoryDB(":memory:")
    conv = ConversationStore(db)
    dag = SummaryDAG(db)
    sess = SessionManager(db)
    cfg = {"context_budget": 800, "context_threshold": 0.75,
           "fresh_tail_count": 4, "leaf_chunk_tokens": 200,
           "leaf_target_tokens": 60, "condensed_target_tokens": 90,
           "min_fanout": 3, "max_rounds": 5}
    if config:
        cfg.update(config)
    engine = CompactionEngine(conv, dag, MockLLM(), config=cfg)
    return db, conv, dag, sess, engine


USER = "u1"


class TestResolveFreshTailOrdinal(unittest.TestCase):
    """_resolve_fresh_tail_ordinal：从末尾数 N 条 message，返回最旧那条的 ordinal"""

    def test_all_messages(self):
        """全部是 message：fresh_tail_count=4 时保护最后4条"""
        _, conv, dag, sess, engine = _make_engine({"fresh_tail_count": 4})
        sid = sess.create_session(USER)
        for i in range(6):
            conv.add_message(USER, sid, "user", f"消息{i}")
        items = dag.get_context_items(USER, sid)
        # 6条消息 ordinal 0-5，保护最后4条 → fresh tail 从 ordinal 2 开始
        fresh = engine._resolve_fresh_tail_ordinal(items)
        self.assertEqual(fresh, 2)

    def test_fewer_than_fresh_tail_count(self):
        """消息数少于 fresh_tail_count：所有消息都受保护，compactable zone 为空"""
        _, conv, dag, sess, engine = _make_engine({"fresh_tail_count": 8})
        sid = sess.create_session(USER)
        for i in range(3):
            conv.add_message(USER, sid, "user", f"短消息{i}")
        items = dag.get_context_items(USER, sid)
        fresh = engine._resolve_fresh_tail_ordinal(items)
        # 只有3条消息，fresh tail 应覆盖全部，compactable zone 无可选
        self.assertLessEqual(fresh, items[0]["ordinal"])

    def test_mixed_items_only_messages_count(self):
        """context_items 包含 summary 和 message：fresh tail 只数 message"""
        _, conv, dag, sess, engine = _make_engine({"fresh_tail_count": 2})
        sid = sess.create_session(USER)
        # 添加4条消息
        for i in range(4):
            conv.add_message(USER, sid, "user", f"消息{i}")
        items_before = dag.get_context_items(USER, sid)
        # 用 summary 替换 ordinal 0-1
        s_id = dag.add_summary(USER, sid, depth=0, content="摘要X",
                                source_start_id=items_before[0]["message_id"],
                                source_end_id=items_before[1]["message_id"])
        dag.replace_context_range(USER, sid, 0, 1, s_id)
        # 现在 context_items: [summary(0), msg(1), msg(2)]
        items = dag.get_context_items(USER, sid)
        fresh = engine._resolve_fresh_tail_ordinal(items)
        # fresh_tail_count=2，只数 message，最后2条 message ordinal 是 1,2
        # fresh tail 从 ordinal 1 开始
        self.assertEqual(fresh, 1)

    def test_zero_fresh_tail(self):
        """fresh_tail_count=0：无保护，返回 inf"""
        _, conv, dag, sess, engine = _make_engine({"fresh_tail_count": 0})
        sid = sess.create_session(USER)
        conv.add_message(USER, sid, "user", "消息")
        items = dag.get_context_items(USER, sid)
        fresh = engine._resolve_fresh_tail_ordinal(items)
        self.assertEqual(fresh, float("inf"))


class TestSelectOldestLeafChunk(unittest.TestCase):
    """_select_oldest_leaf_chunk：跳过开头非 message，遇到非 message 立即停止"""

    def test_selects_oldest_messages(self):
        """连续 message 从头开始：选取最旧的连续块"""
        _, conv, dag, sess, engine = _make_engine(
            {"fresh_tail_count": 2, "leaf_chunk_tokens": 9999})
        sid = sess.create_session(USER)
        for i in range(6):
            conv.add_message(USER, sid, "user", f"消息{i}")
        items = dag.get_context_items(USER, sid)
        fresh = engine._resolve_fresh_tail_ordinal(items)
        chunk = engine._select_oldest_leaf_chunk(items, fresh)
        # fresh tail 保护最后2条（ordinal 4,5），可选 ordinal 0-3
        self.assertGreater(len(chunk), 0)
        self.assertTrue(all(it["item_type"] == "message" for it in chunk))
        self.assertEqual(chunk[0]["ordinal"], 0)

    def test_skips_leading_summary(self):
        """开头是 summary：应该跳过，从第一个 message 开始"""
        _, conv, dag, sess, engine = _make_engine(
            {"fresh_tail_count": 2, "leaf_chunk_tokens": 9999})
        sid = sess.create_session(USER)
        for i in range(5):
            conv.add_message(USER, sid, "user", f"消息{i}")
        items_before = dag.get_context_items(USER, sid)
        # 把 ordinal 0 替换成 summary
        s_id = dag.add_summary(USER, sid, depth=0, content="摘要",
                                source_start_id=items_before[0]["message_id"],
                                source_end_id=items_before[0]["message_id"])
        dag.replace_context_range(USER, sid, 0, 0, s_id)
        # 现在: [summary(0), msg(1), msg(2), msg(3), msg(4)]
        items = dag.get_context_items(USER, sid)
        fresh = engine._resolve_fresh_tail_ordinal(items)
        chunk = engine._select_oldest_leaf_chunk(items, fresh)
        # 应该跳过 summary，从 msg(1) 开始
        self.assertGreater(len(chunk), 0)
        self.assertTrue(all(it["item_type"] == "message" for it in chunk))
        self.assertGreater(chunk[0]["ordinal"], 0)

    def test_stops_at_summary_in_middle(self):
        """message 中间夹着 summary：遇到 summary 立即停止"""
        _, conv, dag, sess, engine = _make_engine(
            {"fresh_tail_count": 1, "leaf_chunk_tokens": 9999})
        sid = sess.create_session(USER)
        for i in range(5):
            conv.add_message(USER, sid, "user", f"消息{i}")
        items_before = dag.get_context_items(USER, sid)
        # 把 ordinal 2 替换成 summary，造成 [msg,msg,summary,msg,msg]
        s_id = dag.add_summary(USER, sid, depth=0, content="中间摘要",
                                source_start_id=items_before[2]["message_id"],
                                source_end_id=items_before[2]["message_id"])
        dag.replace_context_range(USER, sid, 2, 2, s_id)
        items = dag.get_context_items(USER, sid)
        fresh = engine._resolve_fresh_tail_ordinal(items)
        chunk = engine._select_oldest_leaf_chunk(items, fresh)
        # 只应该选 ordinal 0,1（遇到 summary@2 停止）
        self.assertEqual(len(chunk), 2)
        self.assertEqual(chunk[0]["ordinal"], 0)
        self.assertEqual(chunk[-1]["ordinal"], 1)

    def test_respects_fresh_tail(self):
        """fresh tail 之内的 message 不可选"""
        _, conv, dag, sess, engine = _make_engine(
            {"fresh_tail_count": 6, "leaf_chunk_tokens": 9999})
        sid = sess.create_session(USER)
        for i in range(6):
            conv.add_message(USER, sid, "user", f"消息{i}")
        items = dag.get_context_items(USER, sid)
        fresh = engine._resolve_fresh_tail_ordinal(items)
        chunk = engine._select_oldest_leaf_chunk(items, fresh)
        # 全部都在 fresh tail 内，没有可选的
        self.assertEqual(len(chunk), 0)


class TestSelectOldestChunkAtDepth(unittest.TestCase):
    """_select_oldest_chunk_at_depth：跳过非目标，一旦开始遇到非目标立即停止"""

    def _add_summaries(self, dag, conv, sess, depths, fresh_tail_count=1):
        """helper：先加若干 message，再把每条 message 替换成对应 depth 的 summary"""
        sid = sess.create_session(USER)
        # 先加足够多的消息垫底（保证 fresh tail 之外有内容）
        conv.add_message(USER, sid, "user", "最新消息（fresh tail）")
        # 按 depths 顺序创建 summary 并插入 context_items
        # 简化：直接用 context_items cursor 手动构建
        cursor = dag.db.conn.cursor()
        for i, depth in enumerate(depths):
            s_id = dag.add_summary(USER, sid, depth=depth,
                                   content=f"摘要depth{depth}_#{i}")
            cursor.execute(
                "INSERT INTO context_items (user_id, session_id, ordinal, item_type, summary_id) "
                "VALUES (?, ?, ?, 'summary', ?)",
                (USER, sid, i, s_id),
            )
        # fresh tail message 放在最后
        msg_items = dag.get_context_items(USER, sid)
        max_ord = max(it["ordinal"] for it in msg_items) if msg_items else -1
        # 更新已有的 message ordinal 到末尾（它在 ordinal=0 被 summary 占了）
        # 实际 add_message 已写入 ordinal 0，summary 也用 ordinal 0，会冲突
        # 改用不同 session 来构造场景
        dag.db.conn.commit()
        return sid

    def test_selects_oldest_chunk_at_target_depth(self):
        """选取最旧的连续同 depth=0 summary chunk"""
        _, conv, dag, sess, engine = _make_engine(
            {"fresh_tail_count": 1, "leaf_chunk_tokens": 9999, "min_fanout": 3})
        sid = sess.create_session(USER)
        # 添加一条消息作为 fresh tail
        conv.add_message(USER, sid, "user", "fresh tail 消息")
        # 直接在 context_items 里构建3个 depth=0 summary + 1个 depth=1
        cursor = dag.db.conn.cursor()
        # 先清掉 add_message 写入的 context_items 条目
        cursor.execute("DELETE FROM context_items WHERE user_id=? AND session_id=?", (USER, sid))

        s_ids = []
        for i in range(3):
            s_id = dag.add_summary(USER, sid, depth=0, content=f"leaf摘要{i}")
            s_ids.append(s_id)
            cursor.execute(
                "INSERT INTO context_items (user_id, session_id, ordinal, item_type, summary_id) "
                "VALUES (?, ?, ?, 'summary', ?)", (USER, sid, i, s_id))
        # 插入一条 message 作为 fresh tail（ordinal=3）
        conv.add_message(USER, sid, "user", "fresh消息")
        # 手动修正 ordinal（add_message 会写 MAX(ordinal)+1=3）
        items = dag.get_context_items(USER, sid)
        dag.db.conn.commit()

        items = dag.get_context_items(USER, sid)
        fresh = engine._resolve_fresh_tail_ordinal(items)
        chunk = engine._select_oldest_chunk_at_depth(items, depth=0, fresh_tail_ordinal=fresh)
        self.assertEqual(len(chunk), 3)
        self.assertTrue(all(it["item_type"] == "summary" for it in chunk))

    def test_skips_leading_different_depth(self):
        """开头是不同 depth 的 summary：应该跳过"""
        _, conv, dag, sess, engine = _make_engine(
            {"fresh_tail_count": 1, "leaf_chunk_tokens": 9999})
        sid = sess.create_session(USER)
        cursor = dag.db.conn.cursor()
        cursor.execute("DELETE FROM context_items WHERE user_id=? AND session_id=?", (USER, sid))

        # ordinal 0: depth=1，ordinal 1,2: depth=0
        s_d1 = dag.add_summary(USER, sid, depth=1, content="condensed")
        s_d0a = dag.add_summary(USER, sid, depth=0, content="leaf A")
        s_d0b = dag.add_summary(USER, sid, depth=0, content="leaf B")
        cursor.execute("INSERT INTO context_items VALUES (NULL,?,?,0,'summary',NULL,?,CURRENT_TIMESTAMP)", (USER, sid, s_d1))
        cursor.execute("INSERT INTO context_items VALUES (NULL,?,?,1,'summary',NULL,?,CURRENT_TIMESTAMP)", (USER, sid, s_d0a))
        cursor.execute("INSERT INTO context_items VALUES (NULL,?,?,2,'summary',NULL,?,CURRENT_TIMESTAMP)", (USER, sid, s_d0b))
        # fresh tail：message 在 ordinal 3
        conv.add_message(USER, sid, "user", "fresh")
        dag.db.conn.commit()

        items = dag.get_context_items(USER, sid)
        fresh = engine._resolve_fresh_tail_ordinal(items)
        chunk = engine._select_oldest_chunk_at_depth(items, depth=0, fresh_tail_ordinal=fresh)
        # 应该跳过 ordinal 0（depth=1），从 ordinal 1 开始选 depth=0
        self.assertEqual(len(chunk), 2)
        self.assertEqual(chunk[0]["summary_id"], s_d0a)

    def test_stops_at_different_depth_in_middle(self):
        """中间夹着不同 depth：遇到后立即停止"""
        _, conv, dag, sess, engine = _make_engine(
            {"fresh_tail_count": 1, "leaf_chunk_tokens": 9999})
        sid = sess.create_session(USER)
        cursor = dag.db.conn.cursor()
        cursor.execute("DELETE FROM context_items WHERE user_id=? AND session_id=?", (USER, sid))

        # ordinal 0,1: depth=0，ordinal 2: depth=1，ordinal 3: depth=0
        s0a = dag.add_summary(USER, sid, depth=0, content="leaf A")
        s0b = dag.add_summary(USER, sid, depth=0, content="leaf B")
        s1  = dag.add_summary(USER, sid, depth=1, content="condensed")
        s0c = dag.add_summary(USER, sid, depth=0, content="leaf C")
        cursor.execute("INSERT INTO context_items VALUES (NULL,?,?,0,'summary',NULL,?,CURRENT_TIMESTAMP)", (USER, sid, s0a))
        cursor.execute("INSERT INTO context_items VALUES (NULL,?,?,1,'summary',NULL,?,CURRENT_TIMESTAMP)", (USER, sid, s0b))
        cursor.execute("INSERT INTO context_items VALUES (NULL,?,?,2,'summary',NULL,?,CURRENT_TIMESTAMP)", (USER, sid, s1))
        cursor.execute("INSERT INTO context_items VALUES (NULL,?,?,3,'summary',NULL,?,CURRENT_TIMESTAMP)", (USER, sid, s0c))
        conv.add_message(USER, sid, "user", "fresh")
        dag.db.conn.commit()

        items = dag.get_context_items(USER, sid)
        fresh = engine._resolve_fresh_tail_ordinal(items)
        chunk = engine._select_oldest_chunk_at_depth(items, depth=0, fresh_tail_ordinal=fresh)
        # 选到 ordinal 0,1，遇到 ordinal 2（depth=1）停止，不选 ordinal 3
        self.assertEqual(len(chunk), 2)
        self.assertEqual(chunk[-1]["summary_id"], s0b)


class TestCompactEndToEnd(unittest.TestCase):
    """端到端压缩：足够多消息后 compact() 应减少 context token"""

    def test_leaf_compaction_reduces_tokens(self):
        """添加足够消息触发 leaf 压缩，token 数应下降"""
        _, conv, dag, sess, engine = _make_engine({
            "context_budget": 200,
            "context_threshold": 0.5,
            "fresh_tail_count": 2,
            "leaf_chunk_tokens": 300,
            "min_fanout": 3,
            "max_rounds": 5,
        })
        sid = sess.create_session(USER)
        # 添加足够多消息超过阈值（200*0.5=100 tokens）
        for i in range(20):
            conv.add_message(USER, sid, "user", f"这是一条关于雪纳瑞健康的问题，编号{i}，内容比较详细")
            conv.add_message(USER, sid, "assistant", f"这是对应的回答，编号{i}，提供了详细的建议")

        tokens_before = dag.get_context_token_count(USER, sid)
        self.assertGreater(tokens_before, 100)

        stats = engine.compact(USER, sid)
        tokens_after = dag.get_context_token_count(USER, sid)

        self.assertGreater(stats["leaf_count"], 0)
        self.assertLess(tokens_after, tokens_before)

    def test_compact_preserves_fresh_tail(self):
        """压缩后 fresh tail 内的消息仍在 context_items 里"""
        _, conv, dag, sess, engine = _make_engine({
            "context_budget": 100,
            "context_threshold": 0.5,
            "fresh_tail_count": 3,
            "leaf_chunk_tokens": 300,
            "min_fanout": 3,
            "max_rounds": 5,
        })
        sid = sess.create_session(USER)
        for i in range(15):
            conv.add_message(USER, sid, "user", f"问题{i}，较长内容来确保超出 token 阈值")

        # 记录最后3条 message 的 id
        all_items = dag.get_context_items(USER, sid)
        msg_items = [it for it in all_items if it["item_type"] == "message"]
        last3_ids = {it["message_id"] for it in msg_items[-3:]}

        engine.compact(USER, sid)

        after_items = dag.get_context_items(USER, sid)
        after_msg_ids = {it["message_id"] for it in after_items if it["item_type"] == "message"}
        # fresh tail 里的3条消息必须还在
        self.assertTrue(last3_ids.issubset(after_msg_ids))


if __name__ == "__main__":
    unittest.main(verbosity=2)
