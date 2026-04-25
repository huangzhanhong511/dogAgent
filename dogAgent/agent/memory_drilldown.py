"""
dogAgent DAG 记忆钻取模块（Lossless Claw 回溯）

LLM 读取 XML 格式摘要后，通过 memory_expand tool 按 summary_id 触发钻取。
每次调用只返回直接子节点（一层），LLM 自行决定是否继续展开子摘要。

架构位置：
  Condensed Summary (depth N)  ← <summary id="..."> 注入 prompt
       ↓ memory_expand(id)
  Sub-summary (depth N-1)      ← 返回子摘要内容 + id（LLM 可再次展开）
       ↓ memory_expand(id)
  原始消息 (conversations)     ← 无损原始数据
"""

import logging

logger = logging.getLogger("drilldown")


def _estimate_tokens(text):
    cn = sum(1 for c in text if '一' <= c <= '鿿')
    return int(cn / 1.5 + (len(text) - cn) / 4)


class MemoryDrillDown:
    """DAG 记忆钻取引擎（逐层展开，LLM 控制深度）。"""

    def __init__(self, max_drilldown_tokens=2000, max_depth=3):
        self.max_drilldown_tokens = max_drilldown_tokens
        self.max_depth = max_depth

    def drilldown_by_id(self, summary_id: str, user_id: str, summary_dag, conv_store) -> str:
        """
        展开一个摘要的直接子节点（一层）。

        - 子节点是摘要：返回内容 + id，LLM 可再次调用 memory_expand 继续展开
        - 子节点是原始消息：返回格式化消息文本
        """
        children = summary_dag.drill_down(summary_id, user_id, conv_store)
        if not children:
            return "（该摘要已是最底层，无更多子节点）"

        parts = []
        used_tokens = 0

        for child in children:
            if used_tokens >= self.max_drilldown_tokens:
                break

            if child["type"] == "summary":
                hint = f'（如需更多细节，可调用 memory_expand("{child["id"]}")）'
                block = f'<summary id="{child["id"]}">\n{child["content"]}\n</summary>\n{hint}'
                t = _estimate_tokens(block)
                if used_tokens + t <= self.max_drilldown_tokens:
                    parts.append(block)
                    used_tokens += t

            elif child["type"] == "messages":
                budget = self.max_drilldown_tokens - used_tokens
                text = self._format_messages(child["messages"])
                if _estimate_tokens(text) > budget:
                    text = self._truncate_messages(child["messages"], budget)
                    if text:
                        parts.append(f"[原始对话 #{child['start_id']}-#{child['end_id']}（部分）]\n{text}")
                        used_tokens += _estimate_tokens(text)
                else:
                    parts.append(f"[原始对话 #{child['start_id']}-#{child['end_id']}]\n{text}")
                    used_tokens += _estimate_tokens(text)

        return "\n\n".join(parts) if parts else "（未找到子节点内容）"

    def _format_messages(self, messages):
        lines = []
        for msg in messages:
            role = "用户" if msg["role"] == "user" else "助手"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    def _truncate_messages(self, messages, max_tokens):
        lines = []
        used = 0
        for msg in messages:
            role = "用户" if msg["role"] == "user" else "助手"
            line = f"{role}: {msg['content']}"
            t = _estimate_tokens(line)
            if used + t > max_tokens:
                remaining_chars = int((max_tokens - used) * 2)
                if remaining_chars > 20:
                    lines.append(f"{role}: {msg['content'][:remaining_chars]}...")
                lines.append("(... 更多历史消息已截断)")
                break
            lines.append(line)
            used += t
        return "\n".join(lines) if lines else ""
