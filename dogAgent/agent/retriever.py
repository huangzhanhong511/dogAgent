"""
多维度 Wiki 检索器

根据用户查询，通过多维度索引匹配最相关的 Wiki 文章，
只加载必要的文章内容注入 Prompt。

检索维度及权重：
  - 关键词匹配 (0.4)
  - 分类匹配   (0.2)
  - 标签匹配   (0.2)
  - 关联图谱   (0.1)
  - 章节匹配   (0.1)

用法:
    from agent.retriever import WikiRetriever
    r = WikiRetriever()
    results = r.retrieve("雪纳瑞容易得什么病", top_k=3)
"""

import os
import re
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("retriever")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIKI_DIR = os.path.join(PROJECT_DIR, "wiki")
INDEX_PATH = os.path.join(WIKI_DIR, "index.json")

# 查询意图 → 分类映射
INTENT_CATEGORY_MAP = {
    "品种": ["01-品种百科"],
    "breed": ["01-品种百科"],
    "什么狗": ["01-品种百科"],
    "长什么样": ["01-品种百科"],
    "特点": ["01-品种百科"],
    "性格": ["01-品种百科"],

    "吃": ["02-饮食营养"],
    "喂": ["02-饮食营养"],
    "食": ["02-饮食营养"],
    "饮食": ["02-饮食营养"],
    "狗粮": ["02-饮食营养"],
    "营养": ["02-饮食营养"],

    "病": ["03-健康医疗"],
    "症状": ["03-健康医疗"],
    "治疗": ["03-健康医疗"],
    "健康": ["03-健康医疗"],
    "疫苗": ["03-健康医疗"],
    "手术": ["03-健康医疗"],
    "药": ["03-健康医疗"],
    "眼": ["03-健康医疗"],
    "白内障": ["03-健康医疗"],

    "美容": ["04-美容护理"],
    "毛": ["04-美容护理"],
    "剪": ["04-美容护理"],
    "洗澡": ["04-美容护理"],
    "护理": ["04-美容护理"],

    "训练": ["05-训练与行为"],
    "行为": ["05-训练与行为"],
    "叫": ["05-训练与行为"],
    "咬": ["05-训练与行为"],
}

# 查询意图 → 标签映射
INTENT_TAG_MAP = {
    "品种": "品种",
    "什么狗": "品种",
    "性格": "品种",
    "病": "健康医疗",
    "症状": "健康医疗",
    "治疗": "健康医疗",
    "健康": "健康医疗",
    "吃": "饮食营养",
    "喂": "饮食营养",
    "食": "饮食营养",
    "美容": "美容护理",
    "毛": "美容护理",
    "训练": "训练行为",
}


@dataclass
class RetrievalResult:
    """检索结果"""
    title: str
    path: str
    score: float
    match_reasons: list = field(default_factory=list)
    content: str = ""
    sections_matched: list = field(default_factory=list)


class WikiRetriever:
    """多维度 Wiki 检索器"""

    def __init__(self, wiki_dir: str = None, index_path: str = None):
        self.wiki_dir = wiki_dir or WIKI_DIR
        self.index_path = index_path or INDEX_PATH
        self.index = {}
        self._load_index()

    def _load_index(self):
        """加载索引文件"""
        if not os.path.exists(self.index_path):
            logger.warning(f"索引文件不存在: {self.index_path}，请先运行 build_index.py")
            return
        with open(self.index_path, "r", encoding="utf-8") as f:
            self.index = json.load(f)
        logger.info(f"已加载索引: {len(self.index)} 个条目")

    def retrieve(self, query: str, top_k: int = 3) -> list:
        """
        多维度检索

        Args:
            query: 用户查询
            top_k: 返回前 K 个最相关的文章

        Returns:
            list[RetrievalResult]: 按相关度排序的检索结果
        """
        if not self.index:
            logger.warning("索引为空")
            return []

        query_lower = query.lower()
        scores = {}  # title → RetrievalResult

        for title, entry in self.index.items():
            result = RetrievalResult(
                title=title,
                path=entry["path"],
                score=0.0,
            )

            # === 维度 1: 关键词匹配 (权重 0.4) ===
            kw_score = self._score_keywords(query_lower, entry.get("keywords", []))
            if kw_score > 0:
                result.score += kw_score * 0.4
                result.match_reasons.append(f"关键词匹配({kw_score:.2f})")

            # === 维度 2: 分类匹配 (权重 0.2) ===
            cat_score = self._score_category(query_lower, entry.get("category", ""))
            if cat_score > 0:
                result.score += cat_score * 0.2
                result.match_reasons.append(f"分类匹配({cat_score:.2f})")

            # === 维度 3: 标签匹配 (权重 0.2) ===
            tag_score = self._score_tags(query_lower, entry.get("tags", []))
            if tag_score > 0:
                result.score += tag_score * 0.2
                result.match_reasons.append(f"标签匹配({tag_score:.2f})")

            # === 维度 4: 关联匹配 (权重 0.1) ===
            rel_score = self._score_related(query_lower, entry.get("related", []))
            if rel_score > 0:
                result.score += rel_score * 0.1
                result.match_reasons.append(f"关联匹配({rel_score:.2f})")

            # === 维度 5: 章节匹配 (权重 0.1) ===
            sec_score, matched_secs = self._score_sections(query_lower, entry.get("sections", []))
            if sec_score > 0:
                result.score += sec_score * 0.1
                result.sections_matched = matched_secs
                result.match_reasons.append(f"章节匹配({sec_score:.2f}): {matched_secs}")

            # === 标题直接匹配加分 ===
            if query_lower in title.lower() or title.lower() in query_lower:
                result.score += 0.5
                result.match_reasons.append("标题直接匹配(+0.5)")

            if result.score > 0:
                scores[title] = result

        # 排序取 Top-K
        ranked = sorted(scores.values(), key=lambda r: r.score, reverse=True)[:top_k]

        # 加载文章内容
        for result in ranked:
            result.content = self._load_article(result.path, result.sections_matched)

        return ranked

    def _score_keywords(self, query: str, keywords: list) -> float:
        """关键词匹配得分：查询中的词与关键词列表的重合度"""
        if not keywords:
            return 0.0

        # 将查询分词（简单按字符和空格分）
        query_terms = set()
        # 中文按 2-4 字滑窗
        for i in range(len(query)):
            for length in range(2, min(5, len(query) - i + 1)):
                query_terms.add(query[i:i+length])
        # 英文按空格分
        query_terms.update(query.split())

        # 停用词（高频通用词不贡献匹配分）
        stopwords = {"雪纳瑞", "雪纳", "纳瑞", "怎么", "什么", "可以", "能吃", "如何", "为什么", "狗狗", "犬"}

        # 计算匹配
        matches = 0
        matched_kw = []
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower in stopwords:
                continue
            if kw_lower in query or query in kw_lower:
                matches += 3
                matched_kw.append(kw)
            elif any(t in kw_lower for t in query_terms if len(t) >= 2 and t not in stopwords):
                matches += 1
                matched_kw.append(kw)

        if matches == 0:
            return 0.0

        # 按匹配的关键词数量归一化，不受总关键词数影响
        return min(1.0, matches / 5.0)

    def _score_category(self, query: str, category: str) -> float:
        """分类匹配得分"""
        for intent_word, categories in INTENT_CATEGORY_MAP.items():
            if intent_word in query:
                if category in categories:
                    return 1.0
                # 部分匹配（分类名称包含意图关键词）
                if any(intent_word in cat for cat in categories):
                    return 0.5
        return 0.0

    def _score_tags(self, query: str, tags: list) -> float:
        """标签匹配得分"""
        if not tags:
            return 0.0

        matched = 0
        for intent_word, tag in INTENT_TAG_MAP.items():
            if intent_word in query and tag in tags:
                matched += 1

        return min(1.0, matched / max(1, len(tags)))

    def _score_related(self, query: str, related: list) -> float:
        """关联匹配：查询词是否出现在关联条目中"""
        if not related:
            return 0.0
        for rel in related:
            if query in rel.lower() or rel.lower() in query:
                return 1.0
        return 0.0

    def _score_sections(self, query: str, sections: list) -> tuple:
        """章节匹配：查询词是否出现在章节标题中"""
        if not sections:
            return 0.0, []
        matched = []
        for sec in sections:
            sec_lower = sec.lower()
            # 检查查询中的关键词是否在章节标题中
            for intent_word in list(INTENT_CATEGORY_MAP.keys()) + list(INTENT_TAG_MAP.keys()):
                if intent_word in query and intent_word in sec_lower:
                    matched.append(sec)
                    break
            # 直接匹配
            if any(q in sec_lower for q in query.split() if len(q) >= 2):
                if sec not in matched:
                    matched.append(sec)

        score = min(1.0, len(matched) / max(1, len(sections) * 0.3))
        return score, matched

    def _load_article(self, rel_path: str, sections_filter: list = None) -> str:
        """
        加载文章内容

        如果有 sections_filter，只加载匹配的章节（节省 Token）
        否则加载全文
        """
        filepath = os.path.join(self.wiki_dir, rel_path)
        if not os.path.exists(filepath):
            return ""

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # 去掉 frontmatter
        match = re.match(r'^---\s*\n.*?\n---\s*\n(.*)', content, re.DOTALL)
        body = match.group(1).strip() if match else content.strip()

        # 如果有章节过滤且文章较长，只返回匹配章节
        if sections_filter and len(body) > 3000:
            filtered = self._extract_sections(body, sections_filter)
            if filtered:
                return filtered

        return body

    def _extract_sections(self, body: str, section_names: list) -> str:
        """提取指定章节的内容"""
        parts = []
        # 按 H2 分割
        sections = re.split(r'(^## .+$)', body, flags=re.MULTILINE)

        # sections 格式: [前言, "## 标题1", 内容1, "## 标题2", 内容2, ...]
        # 总是包含前言（标题 + 概述）
        if sections:
            parts.append(sections[0].strip())

        i = 1
        while i < len(sections) - 1:
            heading = sections[i].strip()
            content = sections[i + 1].strip() if i + 1 < len(sections) else ""

            # 检查是否匹配过滤条件
            heading_text = heading.replace("## ", "").lower()
            if any(sn.lower() in heading_text or heading_text in sn.lower() for sn in section_names):
                parts.append(f"{heading}\n{content}")

            i += 2

        if len(parts) <= 1:
            # 没有匹配到章节，返回全文
            return body

        return "\n\n".join(parts)

    def format_context(self, results: list) -> str:
        """
        将检索结果格式化为 Prompt 上下文

        Args:
            results: retrieve() 的返回结果

        Returns:
            格式化的上下文字符串
        """
        if not results:
            return "（未找到相关知识库内容）"

        parts = []
        for i, r in enumerate(results, 1):
            parts.append(f"=== 参考文档 {i}: {r.title} (相关度: {r.score:.2f}) ===")
            parts.append(r.content)
            parts.append("")

        return "\n\n".join(parts)

    def explain(self, query: str, top_k: int = 5):
        """调试用：解释检索过程"""
        results = self.retrieve(query, top_k=top_k)
        print(f"\n🔍 查询: \"{query}\"")
        print(f"{'=' * 60}")
        if not results:
            print("  ❌ 没有找到匹配的文章")
            return

        for i, r in enumerate(results, 1):
            print(f"\n  #{i} [{r.score:.3f}] {r.title}")
            print(f"      路径: {r.path}")
            for reason in r.match_reasons:
                print(f"      ✓ {reason}")
            if r.sections_matched:
                print(f"      📄 匹配章节: {r.sections_matched}")
            print(f"      内容长度: {len(r.content)} 字符")


# ============================================================
# WikiVectorStore — 基于 numpy 的轻量向量检索
# ============================================================

VECTOR_DIR = os.path.join(WIKI_DIR, "vectors")
EMBEDDINGS_PATH = os.path.join(VECTOR_DIR, "embeddings.npy")
META_PATH = os.path.join(VECTOR_DIR, "meta.json")


class WikiVectorStore:
    """
    加载 build_vector_index.py 生成的 embeddings.npy + meta.json，
    提供余弦相似度检索。不依赖任何向量数据库，仅用 numpy。
    """

    def __init__(self, embeddings_path: str = None, meta_path: str = None):
        self._embeddings_path = embeddings_path or EMBEDDINGS_PATH
        self._meta_path = meta_path or META_PATH
        self._embeddings: Optional[object] = None  # numpy array [N, D]
        self._meta: list = []
        self._load()

    def _load(self):
        try:
            import numpy as np
            if os.path.exists(self._embeddings_path) and os.path.exists(self._meta_path):
                self._embeddings = np.load(self._embeddings_path)
                with open(self._meta_path, "r", encoding="utf-8") as f:
                    self._meta = json.load(f)
                logger.info(f"向量索引已加载: {self._embeddings.shape[0]} 篇文章")
        except Exception as e:
            logger.warning(f"向量索引加载失败: {e}")

    def is_available(self) -> bool:
        return self._embeddings is not None and len(self._meta) > 0

    def search(
        self,
        query_embedding: list,
        categories: Optional[list] = None,
        top_k: int = 3,
    ) -> list[dict]:
        """
        余弦相似度搜索。

        Args:
            query_embedding: 查询向量（由 embedder.embed_query() 生成）
            categories: 限定搜索的分类列表（如 ["03-健康医疗"]），None 表示全库搜索
            top_k: 返回前 K 个结果

        Returns:
            list of {"title": ..., "path": ..., "category": ..., "score": ...}
        """
        import numpy as np

        q = np.array(query_embedding, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-9)

        # 如果指定分类，只在该分类内搜索
        if categories:
            indices = [
                i for i, m in enumerate(self._meta)
                if m["category"] in categories
            ]
        else:
            indices = list(range(len(self._meta)))

        if not indices:
            return []

        subset = self._embeddings[indices]
        # 归一化每行
        norms = np.linalg.norm(subset, axis=1, keepdims=True) + 1e-9
        subset_norm = subset / norms

        scores = subset_norm @ q  # cosine similarity
        top_local = int(min(top_k, len(indices)))
        top_idx = np.argpartition(scores, -top_local)[-top_local:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        results = []
        for local_i in top_idx:
            global_i = indices[local_i]
            m = self._meta[global_i]
            results.append({
                "title": m["title"],
                "path": m["path"],
                "category": m["category"],
                "score": float(scores[local_i]),
            })
        return results


# ============================================================
# LLMWikiIndexRetriever — Karpathy LLM Wiki 风格检索
# ============================================================


INDEX_MD_PATH = os.path.join(WIKI_DIR, "index.md")

CATEGORY_PROMPT = """你是一个知识库检索助手。以下是雪纳瑞知识库的顶层索引，列出了所有分类及其包含的主题。

请根据用户的问题，判断应该查看哪些分类。返回相关的分类目录名列表。

规则：
1. 只返回 JSON 数组，格式: ["01-品种百科", "03-健康医疗"]
2. 分类目录名必须与索引中完全一致（如 01-品种百科、03-健康医疗）
3. 通常 1-2 个分类就够了，最多 3 个
4. 如果顶层索引中某个分类里直接列出了匹配的文章标题，也把标题放在数组里

---
知识库索引：

{index_content}

---
用户问题：{query}

相关分类："""

ARTICLE_PROMPT = """你是一个知识库检索助手。以下是某分类的文章索引，每条是一篇文章的标题和简要描述。

请根据用户的问题，选择最相关的文章标题（最多 {max_pages} 个）。

规则：
1. 只返回 JSON 数组，格式: ["文章标题1", "文章标题2"]
2. 标题必须与索引中完全一致
3. 如果没有相关文章，返回空数组 []
4. 优先选择最直接相关的文章

---
文章索引：

{index_content}

---
用户问题：{query}

相关文章："""


class LLMWikiIndexRetriever:
    """
    Karpathy LLM Wiki 风格检索器。

    将整个 index.md 注入 LLM prompt，让 LLM 自己判断哪些 Wiki 页面
    与用户问题相关。比规则匹配具有更强的语义理解能力。

    用法：
        retriever = LLMWikiIndexRetriever(llm)
        results = retriever.retrieve("眼睛有白色的东西")
        context = retriever.format_context(results)
    """

    def __init__(self, llm=None, wiki_dir: str = None, index_md_path: str = None):
        self.wiki_dir = wiki_dir or WIKI_DIR
        self.index_md_path = index_md_path or INDEX_MD_PATH
        self.llm = llm
        self.index_content = ""
        self._load_index_md()

        # 保留 WikiRetriever 作为 fallback（LLM 不可用时）
        self._fallback = WikiRetriever(wiki_dir=self.wiki_dir)
        # 向量索引（可选，存在时替代 LLM Step 2）
        self._vector_store = WikiVectorStore()

    @property
    def index(self):
        """兼容 WikiRetriever 的接口（chat.py 会检查 retriever.index）"""
        return self._fallback.index

    def _load_index_md(self):
        """加载 index.md 内容"""
        if os.path.exists(self.index_md_path):
            with open(self.index_md_path, "r", encoding="utf-8") as f:
                self.index_content = f.read()
            logger.info(f"已加载 index.md: {len(self.index_content)} 字符")
        else:
            logger.warning(f"index.md 不存在: {self.index_md_path}，将使用 fallback 检索")

    def retrieve(self, query: str, top_k: int = 3) -> list:
        """
        两层检索：
        Step 1: LLM 读顶层 index.md → 选相关分类
        Step 2: 向量相似度（在选中分类内）→ 选具体文章；向量索引不可用时退回 LLM
        Step 3: 加载文章全文
        """
        if not self.llm or not self.index_content:
            return self._fallback.retrieve(query, top_k=top_k)

        try:
            # Step 1: LLM 读顶层索引 → 选分类或直接选文章
            step1_results = self._llm_select_categories(query)
            if not step1_results:
                logger.info("LLM 未选中任何分类，fallback 到规则检索")
                return self._fallback.retrieve(query, top_k=top_k)

            # 分离：分类目录 vs 直接命中的文章标题
            categories = []
            direct_titles = []
            for item in step1_results:
                if item.startswith("0") and "-" in item[:3]:
                    categories.append(item)
                else:
                    direct_titles.append(item)

            # Step 2: 向量检索（在选中分类内）
            if self._vector_store.is_available() and categories:
                from agent.llm import create_embeddings
                embedder = create_embeddings()
                query_vec = embedder.embed_query(query)
                vec_results = self._vector_store.search(query_vec, categories=categories, top_k=top_k)
                article_paths = [(r["title"], r["path"]) for r in vec_results]
                logger.info(f"向量检索（分类 {categories}）: {[t for t, _ in article_paths]}")
            else:
                # 向量索引不可用，退回 LLM 读分类 index.md
                if not self._vector_store.is_available():
                    logger.info("向量索引不可用，使用 LLM Step 2")
                article_titles = list(direct_titles)
                for cat in categories:
                    cat_index_path = os.path.join(self.wiki_dir, cat, "index.md")
                    if not os.path.exists(cat_index_path):
                        continue
                    with open(cat_index_path, "r", encoding="utf-8") as f:
                        cat_index_content = f.read()
                    titles = self._llm_select_articles(query, cat_index_content, top_k)
                    article_titles.extend(titles)
                article_paths = [
                    (t, self._fallback.index[t]["path"])
                    for t in article_titles
                    if t in self._fallback.index
                ]

            # 补充 direct_titles（Step 1 中顶层索引直接命中的文章）
            for title in direct_titles:
                if title in self._fallback.index:
                    path = self._fallback.index[title]["path"]
                    if (title, path) not in article_paths:
                        article_paths.insert(0, (title, path))

            if not article_paths:
                logger.info("未选中任何文章，fallback 到规则检索")
                return self._fallback.retrieve(query, top_k=top_k)

            # Step 3: 加载选中页面的内容
            results = []
            seen = set()
            for title, path in article_paths:
                if title in seen:
                    continue
                seen.add(title)
                content = self._fallback._load_article(path)
                if content:
                    results.append(RetrievalResult(
                        title=title,
                        path=path,
                        score=1.0,
                        match_reasons=["向量检索" if self._vector_store.is_available() else "LLM index 检索"],
                        content=content,
                    ))
                if len(results) >= top_k:
                    break

            if not results:
                return self._fallback.retrieve(query, top_k=top_k)

            logger.info(f"检索结果: {[r.title for r in results]}")
            return results

        except Exception as e:
            logger.warning(f"LLM 检索失败: {e}，fallback 到规则检索")
            return self._fallback.retrieve(query, top_k=top_k)

    def _llm_select_categories(self, query: str) -> list[str]:
        """Step 1: LLM 读顶层 index.md 选分类"""
        from langchain_core.messages import HumanMessage

        prompt = CATEGORY_PROMPT.format(
            index_content=self.index_content,
            query=query,
        )
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return self._parse_json_list(response)

    def _llm_select_articles(self, query: str, cat_index_content: str, max_pages: int) -> list[str]:
        """Step 2: LLM 读分类 index.md 选文章"""
        from langchain_core.messages import HumanMessage

        prompt = ARTICLE_PROMPT.format(
            max_pages=max_pages,
            index_content=cat_index_content,
            query=query,
        )
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return self._parse_json_list(response)

    def _parse_json_list(self, response) -> list[str]:
        """从 LLM 响应中解析 JSON 数组"""
        raw = response.content if hasattr(response, "content") else str(response)
        raw = raw.strip()

        if raw.startswith("```"):
            raw = re.sub(r"^```\w*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)

        try:
            items = json.loads(raw)
            if isinstance(items, list):
                return [t for t in items if isinstance(t, str)]
        except json.JSONDecodeError:
            logger.warning(f"LLM 返回非 JSON: {raw[:100]}")

        return []

    def format_context(self, results: list) -> str:
        """格式化检索结果为 Prompt 上下文（与 WikiRetriever 兼容）"""
        return self._fallback.format_context(results)

    def explain(self, query: str, top_k: int = 5):
        """调试用：显示 LLM 检索过程"""
        print(f"\n🔍 查询: \"{query}\"")
        print(f"{'=' * 60}")
        print(f"  📄 index.md: {len(self.index_content)} 字符")

        if not self.llm:
            print("  ⚠️  LLM 未设置，使用 fallback")
            self._fallback.explain(query, top_k)
            return

        results = self.retrieve(query, top_k=top_k)
        if not results:
            print("  ❌ 没有找到匹配的文章")
            return

        for i, r in enumerate(results, 1):
            print(f"\n  #{i} {r.title}")
            print(f"      路径: {r.path}")
            for reason in r.match_reasons:
                print(f"      ✓ {reason}")
            print(f"      内容长度: {len(r.content)} 字符")


# ============================================================
# CLI 测试
# ============================================================

if __name__ == "__main__":
    import sys

    retriever = WikiRetriever()

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        retriever.explain(query)
    else:
        # 默认测试几个查询
        test_queries = [
            "雪纳瑞容易得什么病",
            "白内障怎么治疗",
            "标准雪纳瑞的性格特点",
            "迷你雪纳瑞能长多大",
            "雪纳瑞吃什么好",
        ]
        for q in test_queries:
            retriever.explain(q)
            print()