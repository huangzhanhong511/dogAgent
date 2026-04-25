"""
dogAgent 查询重写模块

将依赖上下文的用户查询（如指代消解、省略补全）
重写为独立可检索的完整查询。
"""

import logging

logger = logging.getLogger("query_rewrite")

REWRITE_PROMPT = """你是一个查询重写器。根据最近的对话上下文，将用户的最新问题重写为一个独立的、完整的查询。

规则：
1. 解析代词/指代词（如"它"、"这个"、"那只狗"），替换为具体指称
2. 补全省略信息（如上下文在讨论雪纳瑞，用户问"多久洗一次"，补全为"雪纳瑞多久洗一次澡"）
3. 保持用户原意，不添加新信息
4. 如果查询已经足够完整，原样返回即可
5. 只返回重写后的查询，不要其他文字

最近对话上下文：
{context}

用户最新问题：{query}

重写后的查询："""


class QueryRewriter:
    """查询重写器"""

    def __init__(self, llm=None):
        """
        Args:
            llm: LangChain LLM 实例，为 None 时使用规则重写
        """
        self.llm = llm

    def rewrite(self, query: str, recent_messages: list[dict]) -> str:
        """
        重写查询。

        Args:
            query: 用户原始查询
            recent_messages: 最近的对话消息列表 [{"role": ..., "content": ...}]

        Returns:
            重写后的查询
        """
        # 如果没有上下文或查询已经足够完整，直接返回
        if not recent_messages or len(recent_messages) < 2:
            return query

        # 如果查询较长（>20字）且不含指代词，可能已经完整
        if len(query) > 20 and not self._has_reference(query):
            return query

        # 使用 LLM 重写
        if self.llm:
            return self._rewrite_with_llm(query, recent_messages)

        # Fallback: 规则重写
        return self._rewrite_with_rules(query, recent_messages)

    def _has_reference(self, query: str) -> bool:
        """检测查询中是否有指代词或省略"""
        reference_words = [
            "它", "他", "她", "这个", "那个", "这只", "那只",
            "它的", "他的", "她的", "这种", "那种",
            "上面", "刚才", "前面", "之前",
        ]
        return any(word in query for word in reference_words)

    def _rewrite_with_llm(self, query: str, recent_messages: list[dict]) -> str:
        """使用 LLM 重写查询"""
        # 构建上下文（最近 6 条消息）
        context_msgs = recent_messages[-6:]
        context_lines = []
        for msg in context_msgs:
            role = "用户" if msg["role"] == "user" else "助手"
            content = msg["content"][:300]  # 截断
            context_lines.append(f"{role}：{content}")

        context = "\n".join(context_lines)
        prompt = REWRITE_PROMPT.format(context=context, query=query)

        try:
            response = self.llm.invoke(prompt)
            rewritten = response.content if hasattr(response, 'content') else str(response)
            rewritten = rewritten.strip()

            if rewritten and len(rewritten) < 200:
                logger.info(f"查询重写: '{query}' → '{rewritten}'")
                return rewritten
            else:
                logger.warning(f"LLM 重写结果异常，使用原始查询")
                return query

        except Exception as e:
            logger.error(f"查询重写失败: {e}")
            return query

    def _rewrite_with_rules(self, query: str, recent_messages: list[dict]) -> str:
        """规则重写（Fallback）"""
        # 从最近的用户消息中提取主题
        subject = self._extract_subject(recent_messages)
        if not subject:
            return query

        # 简单替换指代词
        rewritten = query
        for ref in ["它", "这只狗", "那只狗", "这个", "那个"]:
            if ref in rewritten:
                rewritten = rewritten.replace(ref, subject, 1)

        if rewritten != query:
            logger.info(f"规则重写: '{query}' → '{rewritten}'")

        return rewritten

    def _extract_subject(self, recent_messages: list[dict]) -> str:
        """从最近对话中提取主题（简单实现）"""
        # 常见的狗品种关键词
        breed_keywords = [
            "雪纳瑞", "迷你雪纳瑞", "标准雪纳瑞", "巨型雪纳瑞",
            "金毛", "拉布拉多", "泰迪", "柯基", "哈士奇", "柴犬",
            "贵宾", "比熊", "博美", "萨摩耶", "边牧", "德牧",
        ]

        for msg in reversed(recent_messages):
            content = msg["content"]
            for breed in breed_keywords:
                if breed in content:
                    return breed

        return ""