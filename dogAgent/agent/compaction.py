"""
dogAgent DAG 压缩引擎（LCM context_items 版）

压缩流程：
1. Leaf 压缩：选取 context_items 中最旧的连续 message chunk（fresh tail 之外），
   调用 LLM 生成 leaf summary，原地替换 context_items 中的该范围。
2. Condensation：选取 context_items 中最旧的连续同 depth summary chunk，
   当 chunk 数量 >= min_fanout 时合并为更高层 summary，原地替换。
3. 循环直到 context token 降至阈值以下。

严格遵循 LCM selectOldestChunkAtDepth 逻辑：
  - 跳过开头的非目标条目（continue）
  - 一旦开始收集，遇到非目标条目立即停止（break）
  - 不越过 fresh tail 边界
"""

import logging

from agent.memory import _estimate_tokens

logger = logging.getLogger("compaction")

DEFAULT_CONFIG = {
    "context_budget": 8000,
    "context_threshold": 0.75,
    "fresh_tail_count": 8,
    "leaf_chunk_tokens": 4000,
    "leaf_target_tokens": 600,
    "condensed_target_tokens": 900,
    "min_fanout": 6,
    "max_rounds": 5,
}

LEAF_SUMMARY_PROMPT = """请将以下对话内容压缩为一段简洁的摘要。

要求：
1. 保留所有关键信息（诊断、建议、用药、数据等）
2. 保留时间线和因果关系
3. 目标长度约 {target_tokens} 个 token
4. 使用第三人称描述
5. 不要添加对话中没有的信息
6. 最后一行必须是：Expand for details about: <简短列举被省略的具体细节，如确切药名、剂量数值、完整步骤、原始数据等>

对话内容：
{messages}

摘要："""

CONDENSATION_PROMPT = """请将以下多段摘要浓缩为一段更简洁的综合摘要。

要求：
1. 合并重复信息，保留所有独特的关键信息
2. 保持时间线顺序
3. 目标长度约 {target_tokens} 个 token
4. 使用第三人称描述
5. 最后一行必须是：Expand for details about: <简短列举被省略的具体细节>

摘要段落：
{summaries}

综合摘要："""


class CompactionEngine:
    """DAG 压缩引擎（基于 context_items 的 LCM 实现）"""

    def __init__(self, conversation_store, summary_dag, llm, config: dict = None):
        self.conv = conversation_store
        self.dag = summary_dag
        self.llm = llm
        self.config = {**DEFAULT_CONFIG, **(config or {})}

    def should_compact(self, user_id: str, session_id: str) -> bool:
        """判断是否需要压缩（基于 context_items token 总量）"""
        total_tokens = self.dag.get_context_token_count(user_id, session_id)
        threshold = self.config["context_budget"] * self.config["context_threshold"]
        needs = total_tokens > threshold
        if needs:
            logger.info(f"需要压缩: {total_tokens} tokens > {threshold} threshold")
        return needs

    def compact(self, user_id: str, session_id: str) -> dict:
        """执行压缩流程"""
        stats = {"leaf_count": 0, "condensed_count": 0, "total_rounds": 0}

        for round_num in range(self.config["max_rounds"]):
            stats["total_rounds"] = round_num + 1

            leaf_created = self._compact_leaves(user_id, session_id)
            stats["leaf_count"] += leaf_created

            condensed_created = self._condense_all_levels(user_id, session_id)
            stats["condensed_count"] += condensed_created

            if not self.should_compact(user_id, session_id):
                break
            if leaf_created == 0 and condensed_created == 0:
                break

        logger.info(f"压缩完成: {stats}")
        return stats

    def check_and_compact(self, user_id: str, session_id: str) -> dict | None:
        """检查是否需要压缩，需要则执行"""
        if self.should_compact(user_id, session_id):
            return self.compact(user_id, session_id)
        return None

    # ── Private: fresh tail ──────────────────────────────────────────────────

    def _resolve_fresh_tail_ordinal(self, context_items: list[dict]) -> float:
        """
        计算 fresh tail 起始 ordinal（该 ordinal 及之后的原始消息受保护）。
        向前数 fresh_tail_count 条 message 条目，返回最旧那条的 ordinal。
        """
        fresh_tail_count = self.config["fresh_tail_count"]
        if fresh_tail_count <= 0:
            return float("inf")

        msg_items = [item for item in context_items if item["item_type"] == "message"]
        if not msg_items:
            return float("inf")

        protected = 0
        tail_start_ordinal: float = float("inf")
        for item in reversed(msg_items):
            if protected >= fresh_tail_count:
                break
            tail_start_ordinal = item["ordinal"]
            protected += 1

        return tail_start_ordinal

    # ── Private: leaf chunk selection ────────────────────────────────────────

    def _select_oldest_leaf_chunk(
        self, context_items: list[dict], fresh_tail_ordinal: float
    ) -> list[dict]:
        """
        从 context_items 中选取最旧的连续 message chunk（fresh tail 之外）。
        严格遵循 LCM selectOldestLeafChunk 逻辑：
          - 跳过开头的非 message 条目
          - 一旦开始收集，遇到非 message 立即停止
          - 按 leaf_chunk_tokens 限制 chunk 大小
        """
        threshold = self.config["leaf_chunk_tokens"]
        chunk: list[dict] = []
        chunk_tokens = 0
        started = False

        for item in context_items:
            if item["ordinal"] >= fresh_tail_ordinal:
                break

            if not started:
                if item["item_type"] != "message":
                    continue
                started = True
            elif item["item_type"] != "message":
                break

            if item.get("message_id") is None:
                continue

            msg_tokens = self._get_message_token_count(item["message_id"])
            if chunk and chunk_tokens + msg_tokens > threshold:
                break

            chunk.append(item)
            chunk_tokens += msg_tokens
            if chunk_tokens >= threshold:
                break

        return chunk

    def _get_message_token_count(self, message_id: int) -> int:
        cursor = self.conv.db.conn.cursor()
        cursor.execute("SELECT token_count FROM conversations WHERE id = ?", (message_id,))
        row = cursor.fetchone()
        return row["token_count"] if row and row["token_count"] else 50

    def _get_message_content(self, message_id: int) -> dict | None:
        cursor = self.conv.db.conn.cursor()
        cursor.execute(
            "SELECT id, role, content, token_count FROM conversations WHERE id = ?",
            (message_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    # ── Private: condensed chunk selection ──────────────────────────────────

    def _select_oldest_chunk_at_depth(
        self, context_items: list[dict], depth: int, fresh_tail_ordinal: float
    ) -> list[dict]:
        """
        从 context_items 中选取最旧的连续同 depth summary chunk。
        严格遵循 LCM selectOldestChunkAtDepth 逻辑：
          - 跳过开头的非目标条目（非 summary 或不同 depth）
          - 一旦开始收集，遇到任何非目标条目立即停止
          - 按 leaf_chunk_tokens 限制 chunk 大小
        """
        threshold = self.config["leaf_chunk_tokens"]
        chunk: list[dict] = []
        chunk_tokens = 0

        for item in context_items:
            if item["ordinal"] >= fresh_tail_ordinal:
                break

            if item["item_type"] != "summary" or item.get("summary_id") is None:
                if chunk:
                    break
                continue

            summary = self.dag.get_summary_by_id(item["summary_id"])
            if not summary:
                if chunk:
                    break
                continue

            if summary["depth"] != depth:
                if chunk:
                    break
                continue

            s_tokens = summary.get("token_count") or _estimate_tokens(summary["content"])
            if chunk and chunk_tokens + s_tokens > threshold:
                break

            chunk.append(item)
            chunk_tokens += s_tokens
            if chunk_tokens >= threshold:
                break

        return chunk

    # ── Private: leaf pass ───────────────────────────────────────────────────

    def _compact_leaves(self, user_id: str, session_id: str) -> int:
        """
        循环执行 leaf 压缩：每轮选取最旧的 message chunk，生成 leaf summary，
        调用 replace_context_range 原地替换，直到没有可压缩的 chunk。
        """
        created = 0
        while True:
            context_items = self.dag.get_context_items(user_id, session_id)
            fresh_tail_ordinal = self._resolve_fresh_tail_ordinal(context_items)
            chunk = self._select_oldest_leaf_chunk(context_items, fresh_tail_ordinal)
            if not chunk:
                break

            ok = self._create_leaf_from_chunk(user_id, session_id, chunk)
            if not ok:
                break
            created += 1

        return created

    def _create_leaf_from_chunk(
        self, user_id: str, session_id: str, chunk: list[dict]
    ) -> bool:
        """生成一个 leaf summary 并替换 context_items 中的对应范围"""
        msgs = []
        for item in chunk:
            msg = self._get_message_content(item["message_id"])
            if msg:
                msgs.append(msg)
        if not msgs:
            return False

        msg_text = self._format_messages(msgs)
        prompt = LEAF_SUMMARY_PROMPT.format(
            target_tokens=self.config["leaf_target_tokens"],
            messages=msg_text,
        )
        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logger.error(f"Leaf 摘要生成失败: {e}")
            return False

        start_id = msgs[0]["id"]
        end_id = msgs[-1]["id"]
        ordinals = [item["ordinal"] for item in chunk]
        start_ord = min(ordinals)
        end_ord = max(ordinals)

        summary_id = self.dag.add_summary(
            user_id=user_id,
            session_id=session_id,
            depth=0,
            content=content,
            source_start_id=start_id,
            source_end_id=end_id,
            child_ids=[f"msg_range:{start_id}-{end_id}"],
            child_types=["message_range"],
        )
        self.dag.replace_context_range(user_id, session_id, start_ord, end_ord, summary_id)
        logger.info(f"Leaf 摘要: 消息 #{start_id}-#{end_id} → {_estimate_tokens(content)} tokens")
        return True

    # ── Private: condensed pass ──────────────────────────────────────────────

    def _condense_all_levels(self, user_id: str, session_id: str) -> int:
        """逐层（从浅到深）检查并执行 condensation"""
        total = 0
        for depth in self.dag.get_distinct_depths_in_context(user_id, session_id):
            context_items = self.dag.get_context_items(user_id, session_id)
            fresh_tail_ordinal = self._resolve_fresh_tail_ordinal(context_items)
            chunk = self._select_oldest_chunk_at_depth(context_items, depth, fresh_tail_ordinal)
            if len(chunk) >= self.config["min_fanout"]:
                created = self._condense_chunk(user_id, session_id, depth, chunk)
                total += created
        return total

    def _condense_chunk(
        self, user_id: str, session_id: str, depth: int, chunk: list[dict]
    ) -> int:
        """将 chunk 中的 summary 条目浓缩为一个更高层 summary，原地替换 context_items"""
        summaries = []
        child_ids = []
        for item in chunk:
            s = self.dag.get_summary_by_id(item["summary_id"])
            if s:
                summaries.append(s)
                child_ids.append(s["id"])

        if not summaries:
            return 0

        summary_texts = []
        for i, s in enumerate(summaries, 1):
            summary_texts.append(f"--- 摘要 {i} ---\n{s['content']}")
        combined = "\n\n".join(summary_texts)

        prompt = CONDENSATION_PROMPT.format(
            target_tokens=self.config["condensed_target_tokens"],
            summaries=combined,
        )
        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logger.error(f"Condensation 失败 (depth {depth}): {e}")
            return 0

        ordinals = [item["ordinal"] for item in chunk]
        start_ord = min(ordinals)
        end_ord = max(ordinals)
        new_depth = depth + 1

        summary_id = self.dag.add_summary(
            user_id=user_id,
            session_id=session_id,
            depth=new_depth,
            content=content,
            child_ids=child_ids,
            child_types=["summary"] * len(child_ids),
        )
        self.dag.replace_context_range(user_id, session_id, start_ord, end_ord, summary_id)
        logger.info(
            f"Condensation: depth {depth} ({len(summaries)} 个) → depth {new_depth} "
            f"({_estimate_tokens(content)} tokens)"
        )
        return 1

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _format_messages(self, messages: list[dict]) -> str:
        lines = []
        for msg in messages:
            role = "用户" if msg["role"] == "user" else "助手"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)
