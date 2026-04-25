"""
Wiki 索引生成器

扫描 wiki/ 目录下所有 .md 文件，为每篇文章提取多维度索引信息，
输出 wiki/index.json 供 retriever 使用。

索引维度：
  - category: 分类目录
  - tags: 标签列表
  - keywords: 关键词（中英文）
  - sections: H2 章节标题列表
  - related: 双向链接提到的其他条目
  - summary: 首段或概述
  - path: 文件路径

用法:
    python agent/build_index.py
"""

import os
import re
import json
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("build_index")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIKI_DIR = os.path.join(PROJECT_DIR, "wiki")
INDEX_PATH = os.path.join(WIKI_DIR, "index.json")

# 关键词扩展：领域词汇映射，方便从用户查询到文章的匹配
DOMAIN_SYNONYMS = {
    "白内障": ["cataract", "cataracts", "眼睛", "失明", "视力"],
    "标准雪纳瑞": ["standard schnauzer", "标准型", "标雪"],
    "迷你雪纳瑞": ["miniature schnauzer", "迷你型", "迷雪", "mini schnauzer"],
    "巨型雪纳瑞": ["giant schnauzer", "巨型", "巨雪"],
    "胰腺炎": ["pancreatitis", "胰脏", "胰腺"],
    "膀胱结石": ["bladder stones", "尿路结石", "urolithiasis", "结石"],
    "耳部感染": ["ear infection", "otitis", "中耳炎", "外耳炎", "耳道"],
    "皮肤过敏": ["skin allergy", "dermatitis", "特应性皮炎", "atopic", "过敏"],
    "甲状腺功能减退": ["hypothyroidism", "甲减", "甲状腺"],
    "髌骨脱位": ["patellar luxation", "膝关节", "髌骨"],
    "肝分流": ["liver shunt", "portosystemic shunt", "PSS"],
    "糖尿病": ["diabetes", "血糖", "胰岛素"],
    "库欣综合征": ["cushing", "hyperadrenocorticism", "肾上腺"],
    "癫痫": ["epilepsy", "seizure", "抽搐"],
    "牙齿疾病": ["dental disease", "牙周病", "periodontal", "口腔"],
    "皮肤病": ["skin disease", "dermatitis", "皮肤"],
    "髋关节": ["hip dysplasia", "髋关节发育不良"],
    "喂养": ["feeding", "喂食", "饮食", "食物", "狗粮"],
    "训练": ["training", "教育", "服从"],
    "美容": ["grooming", "毛发", "修剪", "剪毛"],
    "手剥": ["hand stripping", "手拔", "剥毛"],
    "吠叫": ["barking", "叫", "吠"],
    "社会化": ["socialization", "社交"],
    "幼犬": ["puppy", "小狗", "仔犬"],
    "老年犬": ["senior dog", "老龄", "老年"],
    "疫苗": ["vaccine", "vaccination", "免疫", "接种"],
    "驱虫": ["deworming", "寄生虫", "体内虫", "体外虫"],
    "绝育": ["spay", "neuter", "绝育手术", "去势"],
    "禁忌食物": ["toxic foods", "有毒", "中毒", "不能吃"],
}


def parse_frontmatter(content: str) -> dict:
    """解析 YAML frontmatter"""
    match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
    if not match:
        return {}
    fm_text = match.group(1)
    metadata = {}
    for line in fm_text.split("\n"):
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip().strip('"')
            if value.startswith("["):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    pass
            metadata[key] = value
    return metadata


def extract_body(content: str) -> str:
    """提取正文（去掉 frontmatter）"""
    match = re.match(r'^---\s*\n.*?\n---\s*\n(.*)', content, re.DOTALL)
    return match.group(1) if match else content


def extract_sections(body: str) -> list:
    """提取所有 H2 章节标题"""
    return re.findall(r'^## (.+)$', body, re.MULTILINE)


def extract_summary(body: str) -> str:
    """提取摘要：首个 blockquote 或首段文本"""
    # 尝试匹配 > 开头的概述
    bq = re.search(r'^> (.+)$', body, re.MULTILINE)
    if bq:
        return bq.group(1).strip()
    # 取首个非空段落
    for line in body.split("\n"):
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("-") and not line.startswith("|"):
            return line[:200]
    return ""


def extract_wikilinks(body: str) -> list:
    """提取 [[双向链接]] 中的标题"""
    return list(set(re.findall(r'\[\[(.+?)\]\]', body)))


def extract_keywords(title: str, body: str, metadata: dict) -> list:
    """
    提取关键词：
    1. 标题本身
    2. frontmatter 中的 sources
    3. 正文中 **加粗** 的词
    4. H1/H2/H3 标题中的词
    5. 领域同义词扩展
    """
    keywords = set()

    # 标题
    keywords.add(title)

    # 英文标题也加（如果有括号标注）
    en_matches = re.findall(r'[（(]([A-Za-z][A-Za-z\s\-]+)[）)]', body)
    for m in en_matches:
        keywords.add(m.strip().lower())

    # 加粗词
    bold_words = re.findall(r'\*\*(.+?)\*\*', body)
    for w in bold_words:
        if len(w) < 30:
            keywords.add(w)

    # H1-H3 标题
    headings = re.findall(r'^#{1,3}\s+(.+)$', body, re.MULTILINE)
    for h in headings:
        # 清理标题中的 emoji 和符号
        clean_h = re.sub(r'[^\w\u4e00-\u9fff\s]', '', h).strip()
        if clean_h:
            keywords.add(clean_h)

    # 领域同义词扩展
    all_text = (title + " " + body).lower()
    for term, synonyms in DOMAIN_SYNONYMS.items():
        if term.lower() in all_text or any(s in all_text for s in synonyms):
            keywords.add(term)
            keywords.update(synonyms)

    # 来源
    sources = metadata.get("sources", [])
    if isinstance(sources, str):
        try:
            sources = json.loads(sources)
        except:
            sources = [sources]
    keywords.update(sources)

    return sorted(k for k in keywords if k and len(k) > 1)


def build_index(wiki_dir: str) -> dict:
    """构建完整索引"""
    index = {}
    all_titles = []

    for root, dirs, files in os.walk(wiki_dir):
        dirs[:] = [d for d in dirs if not d.startswith("_")]
        for filename in files:
            if not filename.endswith(".md") or filename == "index.json":
                continue

            filepath = os.path.join(root, filename)
            rel_path = os.path.relpath(filepath, wiki_dir)

            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            metadata = parse_frontmatter(content)
            body = extract_body(content)
            title = metadata.get("title", filename.replace(".md", "").replace("-", " "))
            category = os.path.basename(os.path.dirname(filepath))

            # 提取各维度
            sections = extract_sections(body)
            summary = extract_summary(body)
            related = extract_wikilinks(body)
            keywords = extract_keywords(title, body, metadata)

            # 标签
            tags = []
            if "品种" in category:
                tags.append("品种")
            if "健康" in category or "医疗" in category:
                tags.append("健康医疗")
            if "饮食" in category or "营养" in category:
                tags.append("饮食营养")
            if "美容" in category or "护理" in category:
                tags.append("美容护理")
            if "训练" in category or "行为" in category:
                tags.append("训练行为")
            if "繁殖" in category or "幼犬" in category:
                tags.append("繁殖幼犬")

            # 从内容推断额外标签
            if any(w in body for w in ["症状", "治疗", "诊断", "disease", "condition"]):
                if "健康医疗" not in tags:
                    tags.append("健康医疗")
            if any(w in body for w in ["喂食", "饮食", "食物", "feeding"]):
                if "饮食营养" not in tags:
                    tags.append("饮食营养")

            sources = metadata.get("sources", [])
            if isinstance(sources, str):
                try:
                    sources = json.loads(sources)
                except:
                    sources = [sources]

            entry = {
                "title": title,
                "path": rel_path,
                "category": category,
                "tags": tags,
                "keywords": keywords,
                "sections": sections,
                "related": related,
                "summary": summary,
                "sources": sources,
                "has_conflicts": metadata.get("has_conflicts") == "true",
                "char_count": len(body),
            }

            index[title] = entry
            all_titles.append(title)
            logger.info(f"  索引: {title} ({category}) - {len(keywords)} 关键词, {len(sections)} 章节")

    return index


def generate_index_md(index: dict, wiki_dir: str):
    """
    生成两层索引（Karpathy LLM Wiki 风格）。

    第 1 层：wiki/index.md — 精简顶层目录，每个分类只列标题，~3-5KB
    第 2 层：wiki/{category}/index.md — 分类级索引，每篇文章标题 + 一句话摘要
    """
    by_category = {}
    for title, entry in index.items():
        cat = entry.get("category", "其他")
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append((title, entry))

    category_names = {
        "01-品种百科": "品种百科",
        "02-饮食营养": "饮食营养",
        "03-健康医疗": "健康医疗",
        "04-美容护理": "美容护理",
        "05-训练与行为": "训练与行为",
        "06-日常饲养": "日常饲养",
        "07-繁殖与幼犬": "繁殖与幼犬",
        "08-法规与养犬常识": "法规与养犬常识",
        "09-参考资料": "参考资料",
    }

    # ── 第 1 层：顶层 index.md（精简版） ──
    top_lines = ["# 雪纳瑞知识库索引\n"]

    for cat_key in sorted(by_category.keys()):
        cat_name = category_names.get(cat_key, cat_key)
        entries = by_category[cat_key]
        titles = sorted([t for t, _ in entries])
        count = len(titles)

        top_lines.append(f"\n## {cat_name}（{count} 篇）")
        if count <= 15:
            top_lines.append("、".join(titles))
        else:
            top_lines.append("、".join(titles[:12]) + f"等（共 {count} 篇）")
            top_lines.append(f"→ 详见 {cat_key}/index.md")

    top_md = "\n".join(top_lines) + "\n"
    top_path = os.path.join(wiki_dir, "index.md")
    with open(top_path, "w", encoding="utf-8") as f:
        f.write(top_md)
    logger.info(f"顶层 index.md: {len(top_md)} 字节, {len(index)} 条目")

    # ── 第 2 层：分类级 index.md ──
    for cat_key, entries in by_category.items():
        cat_name = category_names.get(cat_key, cat_key)
        cat_dir = os.path.join(wiki_dir, cat_key)
        os.makedirs(cat_dir, exist_ok=True)

        cat_lines = [f"# {cat_name}索引\n"]
        for title, entry in sorted(entries, key=lambda x: x[0]):
            summary = entry.get("summary", "")
            if summary and len(summary) > 50:
                summary = summary[:50] + "..."
            if summary:
                cat_lines.append(f"- **{title}** — {summary}")
            else:
                cat_lines.append(f"- **{title}**")

        cat_md = "\n".join(cat_lines) + "\n"
        cat_path = os.path.join(cat_dir, "index.md")
        with open(cat_path, "w", encoding="utf-8") as f:
            f.write(cat_md)
        logger.info(f"  {cat_key}/index.md: {len(cat_md)} 字节, {len(entries)} 条目")

    return top_path


def main():
    logger.info("===== Wiki 索引生成 =====")
    logger.info(f"Wiki 目录: {WIKI_DIR}")

    index = build_index(WIKI_DIR)

    if not index:
        logger.warning("没有找到 Wiki 文章，请先运行 build_wiki.py")
        return

    # 写入 index.json
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    logger.info(f"\n索引已生成: {INDEX_PATH}")
    logger.info(f"共 {len(index)} 个条目")

    # 生成 index.md（Karpathy LLM Wiki 风格，供 LLM 直接阅读）
    generate_index_md(index, WIKI_DIR)

    # 打印索引概览
    for title, entry in index.items():
        logger.info(f"  [{entry['category']}] {title}")
        logger.info(f"    标签: {entry['tags']}")
        logger.info(f"    关键词数: {len(entry['keywords'])}")
        logger.info(f"    章节: {entry['sections'][:5]}{'...' if len(entry['sections']) > 5 else ''}")


if __name__ == "__main__":
    main()