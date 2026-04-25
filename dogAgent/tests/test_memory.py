#!/usr/bin/env python3
"""dogAgent 记忆模块功能测试"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.memory import MemoryDB, ConversationStore, UserPreferences, SummaryDAG
from agent.session import SessionManager


def main():
    db = MemoryDB(":memory:")
    conv = ConversationStore(db)
    prefs = UserPreferences(db)
    dag = SummaryDAG(db)
    sess = SessionManager(db)

    # ---------- 1. Session 管理 ----------
    sid = sess.create_session("u1", "测试会话")
    assert sid
    s = sess.get_session(sid)
    assert s["title"] == "测试会话"
    print("✅ 1. Session 创建/查询")

    sid2 = sess.ensure_session("u1")
    assert sid2 == sid
    print("✅ 2. ensure_session 复用最近会话")

    # ---------- 2. 对话存储 ----------
    conv.add_message("u1", sid, "user", "雪纳瑞多大可以洗澡?")
    conv.add_message("u1", sid, "assistant", "一般3个月后可以洗澡")
    conv.add_message("u1", sid, "user", "用什么沐浴液?")
    conv.add_message("u1", sid, "assistant", "建议用宠物专用沐浴液")
    recent = conv.get_recent("u1", sid, limit=4)
    assert len(recent) == 4
    assert recent[0]["role"] == "user"
    print("✅ 3. 对话存储与检索")

    # ---------- 3. 用户偏好 ----------
    prefs.add_preference("u1", "用户养了一只迷你雪纳瑞")
    prefs.add_preference("u1", "狗狗今年2岁")
    active = prefs.get_active("u1")
    assert len(active) == 2
    print("✅ 4. 用户偏好设置与查询")

    old_id = [p for p in active if "2岁" in p["content"]][0]["id"]
    prefs.update_preference(old_id, "狗狗今年3岁了", "u1")
    active2 = prefs.get_active("u1")
    age_pref = [p for p in active2 if "3岁" in p["content"]]
    assert len(age_pref) == 1
    print("✅ 5. 偏好级联更新")

    # ---------- 4. DAG 摘要 ----------
    s_id = dag.add_summary(
        "u1", sid, depth=0,
        content="用户询问了雪纳瑞洗澡和沐浴液",
        source_start_id=1, source_end_id=4,
    )
    s = dag.get_summary_by_id(s_id)
    assert s is not None
    assert "洗澡" in s["content"]
    print("✅ 6. DAG 摘要存储与检索")

    # ---------- 5. 消息数量 ----------
    msg_count = sess.get_message_count(sid)
    assert msg_count == 4
    print(f"✅ 7. 会话消息数: {msg_count}")

    # ---------- 6. 多租户隔离 ----------
    sid_b = sess.create_session("u2", "用户B会话")
    conv.add_message("u2", sid_b, "user", "巨型雪纳瑞有多大?")
    recent_b = conv.get_recent("u2", sid_b, limit=10)
    recent_a = conv.get_recent("u1", sid, limit=10)
    assert len(recent_b) == 1
    assert len(recent_a) == 4
    print("✅ 8. 多租户数据隔离")

    # ---------- 7. 会话归档 ----------
    sid3 = sess.create_session("u1", "第二个会话")
    sess.archive_session(sid)
    latest = sess.get_latest_session("u1")
    assert latest is not None
    assert latest["id"] == sid3
    print("✅ 9. 会话归档")

    # ---------- 8. 会话列表 ----------
    sessions = sess.list_sessions("u1")
    assert len(sessions) >= 1
    print(f"✅ 10. 会话列表: {len(sessions)} 个")

    # ---------- 11. DAG 回溯（Lossless Claw drill-down） ----------
    leaf_id = dag.add_summary(
        "u1", sid3, depth=0,
        content="用户询问了雪纳瑞耳道感染的治疗方案，助手建议使用耳肤灵",
        source_start_id=1, source_end_id=4,
        child_ids=["msg_range:1-4"],
        child_types=["message_range"],
    )
    children = dag.drill_down(leaf_id, "u1", conv)
    assert len(children) > 0
    assert children[0]["type"] == "messages"
    assert len(children[0]["messages"]) > 0
    print("✅ 11. DAG drill_down 回溯到原始消息")

    # ---------- 12. MemoryDrillDown 引擎 ----------
    from agent.memory_drilldown import MemoryDrillDown
    dd = MemoryDrillDown(max_drilldown_tokens=2000)
    result = dd.drilldown_by_id(leaf_id, "u1", dag, conv)
    assert isinstance(result, str)
    print("✅ 12. MemoryDrillDown drilldown_by_id 执行")

    # ---------- 13. 偏好 Decay ----------
    prefs.add_preference("u1", "旺旺喜欢吃鸡肉")
    active = prefs.get_active("u1")
    chicken_pref = [p for p in active if "鸡肉" in p["content"]][0]
    assert chicken_pref.get("last_confirmed_at") is not None
    print("✅ 13. 偏好 last_confirmed_at 初始化")

    assert not prefs._is_stale(chicken_pref)
    print("✅ 14. 新鲜偏好不过时")

    import sqlite3
    db.conn.cursor().execute(
        "UPDATE user_preferences SET last_confirmed_at = datetime('now', '-60 days') WHERE id = ?",
        (chicken_pref["id"],),
    )
    db.conn.commit()
    stale_active = prefs.get_active("u1")
    stale_pref = [p for p in stale_active if "鸡肉" in p["content"]][0]
    assert prefs._is_stale(stale_pref)
    print("✅ 15. 过时偏好被正确标记")

    text = prefs.get_active_text("u1")
    assert "可能已过时" in text
    assert "鸡肉" in text
    print("✅ 16. 过时偏好在 prompt 文本中被标注")

    prefs.touch_preferences("u1")
    refreshed = prefs.get_active("u1")
    refreshed_pref = [p for p in refreshed if "鸡肉" in p["content"]][0]
    assert not prefs._is_stale(refreshed_pref)
    print("✅ 17. touch_preferences 刷新确认时间")

    # ========== LCM context_items 专项测试 ==========

    sid_lcm = sess.create_session("u1", "LCM测试会话")

    # ---------- 14. add_message 同步写入 context_items ----------
    conv.add_message("u1", sid_lcm, "user", "旺旺今天吃了什么？")
    conv.add_message("u1", sid_lcm, "assistant", "旺旺今天吃了鸡肉和蔬菜")
    conv.add_message("u1", sid_lcm, "user", "明天吃什么好？")
    items = dag.get_context_items("u1", sid_lcm)
    assert len(items) == 3
    assert all(it["item_type"] == "message" for it in items)
    # ordinal 严格连续 0,1,2
    assert [it["ordinal"] for it in items] == [0, 1, 2]
    print("✅ 18. add_message 自动写入 context_items，ordinal 连续")

    # ---------- 15. get_context_token_count ----------
    token_count = dag.get_context_token_count("u1", sid_lcm)
    assert token_count > 0
    print(f"✅ 19. get_context_token_count: {token_count} tokens")

    # ---------- 16. replace_context_range 原子替换 ----------
    conv.add_message("u1", sid_lcm, "assistant", "明天可以吃牛肉")
    # 现在有 ordinals 0,1,2,3 共4条消息
    items_before = dag.get_context_items("u1", sid_lcm)
    assert len(items_before) == 4

    # 用一个 leaf summary 替换 ordinal 0-1
    msg_ids = [it["message_id"] for it in items_before if it["ordinal"] in (0, 1)]
    summary_id = dag.add_summary(
        "u1", sid_lcm, depth=0,
        content="用户询问了旺旺的饮食，助手回答了鸡肉和蔬菜",
        source_start_id=msg_ids[0], source_end_id=msg_ids[1],
        child_ids=[f"msg_range:{msg_ids[0]}-{msg_ids[1]}"],
        child_types=["message_range"],
    )
    dag.replace_context_range("u1", sid_lcm, 0, 1, summary_id)

    items_after = dag.get_context_items("u1", sid_lcm)
    # 4条 → 1个summary + 2条消息 = 3条，ordinal 重排为 0,1,2
    assert len(items_after) == 3
    assert items_after[0]["item_type"] == "summary"
    assert items_after[0]["summary_id"] == summary_id
    assert items_after[0]["ordinal"] == 0
    assert [it["ordinal"] for it in items_after] == [0, 1, 2]
    print("✅ 20. replace_context_range 原子替换，ordinal 重排连续")

    # ---------- 17. get_distinct_depths_in_context ----------
    depths = dag.get_distinct_depths_in_context("u1", sid_lcm)
    assert depths == [0]
    print("✅ 21. get_distinct_depths_in_context 返回正确深度列表")

    # 再加一个 depth=1 的 condensed summary
    condensed_id = dag.add_summary(
        "u1", sid_lcm, depth=1,
        content="综合摘要：旺旺饮食问题讨论",
        child_ids=[summary_id],
        child_types=["summary"],
    )
    # 手动插入 context_items（模拟 condensation 后的状态）
    cursor = db.conn.cursor()
    cursor.execute(
        "DELETE FROM context_items WHERE user_id = ? AND session_id = ? AND ordinal = 0",
        ("u1", sid_lcm),
    )
    cursor.execute(
        "INSERT INTO context_items (user_id, session_id, ordinal, item_type, summary_id) VALUES (?, ?, 0, 'summary', ?)",
        ("u1", sid_lcm, condensed_id),
    )
    db.conn.commit()

    depths2 = dag.get_distinct_depths_in_context("u1", sid_lcm)
    assert 1 in depths2
    print("✅ 22. 多层 depth 时 get_distinct_depths_in_context 返回所有层级")

    # ---------- 18. get_context_text 从 context_items 读取（含 XML 格式） ----------
    ctx_text = dag.get_context_text("u1", sid_lcm)
    assert "<summary" in ctx_text
    assert 'depth="1"' in ctx_text
    assert "综合摘要" in ctx_text
    print("✅ 23. get_context_text 从 context_items 读取，输出 XML 格式")

    # ---------- 19. get_context_text 混合深度顺序 ----------
    # context_items 里 ordinal 0 是 depth=1，1和2是 message（不渲染）
    # 只有 summary 类型出现在 context_text 里
    sid_mix = sess.create_session("u1", "混合深度测试")
    leaf1_id = dag.add_summary("u1", sid_mix, depth=0, content="leaf摘要A", source_start_id=1, source_end_id=2)
    leaf2_id = dag.add_summary("u1", sid_mix, depth=0, content="leaf摘要B", source_start_id=3, source_end_id=4)
    cond_id  = dag.add_summary("u1", sid_mix, depth=1, content="condensed摘要X", child_ids=[leaf1_id], child_types=["summary"])

    # 手动构造混合顺序：ordinal 0 = condensed(depth=1)，ordinal 1 = leaf(depth=0)
    cursor = db.conn.cursor()
    cursor.execute("DELETE FROM context_items WHERE user_id = ? AND session_id = ?", ("u1", sid_mix))
    cursor.execute("INSERT INTO context_items (user_id, session_id, ordinal, item_type, summary_id) VALUES ('u1', ?, 0, 'summary', ?)", (sid_mix, cond_id))
    cursor.execute("INSERT INTO context_items (user_id, session_id, ordinal, item_type, summary_id) VALUES ('u1', ?, 1, 'summary', ?)", (sid_mix, leaf2_id))
    db.conn.commit()

    mix_text = dag.get_context_text("u1", sid_mix)
    # condensed 必须在 leaf 之前（按 ordinal 顺序）
    assert mix_text.index("condensed摘要X") < mix_text.index("leaf摘要B")
    assert 'depth="1"' in mix_text
    assert 'depth="0"' in mix_text
    print("✅ 24. get_context_text 按 ordinal 顺序展示混合深度摘要")

    print("\n🎉 所有记忆模块功能测试通过！（含 LCM context_items 专项）")


if __name__ == "__main__":
    main()
