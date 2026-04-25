"""
LLM Wiki 生成器

核心原则：
1. 无损转换 — 保留所有原始信息，不丢弃任何有效内容
2. 多源合并 — 将同一主题的多篇文章合并为一个 Wiki 条目
3. 冲突检测 — 不同来源信息不一致时，记录到冲突报告供人工审核
4. 结构化 — 按模板生成结构化的 Wiki 页面

流程：
  knowledge/{分类}/*.md  →  LLM 处理  →  wiki/{分类}/*.md  +  conflicts/冲突报告.md
"""

import os
import re
import json
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

from langchain_core.messages import HumanMessage, SystemMessage

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("build_wiki")

# 路径
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOWLEDGE_DIR = os.environ.get("KNOWLEDGE_DIR", os.path.join(PROJECT_DIR, "knowledge"))
WIKI_DIR = os.path.join(PROJECT_DIR, "wiki")
CONFLICTS_DIR = os.path.join(PROJECT_DIR, "conflicts")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "qwen-plus")

# ============================================================
# System Prompts
# ============================================================

WIKI_SYSTEM_PROMPT = """你是一个专业的宠物医疗知识编辑，负责将多个来源的原始文章合并生成一个无损的中文 Wiki 条目。

## 核心原则

1. **零信息损失**: 每一条事实、数据、建议都必须保留在最终 Wiki 中。不确定是否重要就保留。
2. **忠实原文**: 翻译英文时保持原意，不添加原文没有的信息。保留关键英文术语（用括号标注）。
3. **多源整合**: 将不同来源的信息整合到统一结构中，注明每条信息的来源。
4. **冲突标注**: 如果不同来源说法不一致，**不要取舍**，全部保留并明确标注冲突。

## 输出格式

请严格按以下 JSON 格式输出（不要 markdown 代码块包裹）：

{
  "wiki_title": "中文 Wiki 标题",
  "wiki_content": "完整的 Markdown 格式 Wiki 正文（见下方结构要求）",
  "conflicts": [
    {
      "topic": "冲突主题",
      "description": "冲突描述",
      "source_a": {"source": "来源A名称", "claim": "来源A的说法"},
      "source_b": {"source": "来源B名称", "claim": "来源B的说法"},
      "severity": "高/中/低"
    }
  ],
  "all_sources_used": ["来源1", "来源2"],
  "information_coverage": "完整/部分缺失",
  "notes_for_reviewer": "给人工审核者的备注（如有）"
}

## Wiki 正文结构要求

Wiki 正文应该是完整的 Markdown 文档，根据主题类型使用以下结构：

### 疾病类 Wiki 结构：
```
# {疾病名称}（{英文名}）

> 一句话概述

## 基本信息
- 常见程度：
- 雪纳瑞易感度：高/中/低
- 紧急程度：

## 概述
（疾病的总体描述）

## 病因
（所有来源提到的病因，标注来源）

## 症状
（所有症状，分轻度/中度/重度如有）

## 诊断
（诊断方法）

## 治疗
（治疗方案，包括药物、手术、饮食调整等）

## 预防
（预防措施）

## 居家护理
（主人可以做的护理）

## 预后
（治疗效果、恢复期等）

## 来源与参考
（列出所有原始来源）
```

### 营养/食物类 Wiki 结构：
```
# {食物/营养主题}

> 一句话概述

## 基本信息
- 安全性：安全/有条件安全/危险/剧毒
- 雪纳瑞特别注意：

## 详细说明

## 营养价值/风险

## 喂食建议（如安全）
- 推荐量：
- 频率：
- 制备方式：

## 危险性说明（如不安全）
- 中毒症状：
- 紧急处理：

## 来源与参考
```

### 护理/美容类 Wiki 结构：
```
# {护理主题}

> 一句话概述

## 雪纳瑞特别说明

## 所需工具

## 步骤详解

## 频率建议

## 常见问题

## 注意事项

## 来源与参考
```

## 冲突检测规则

以下情况视为冲突，必须记录：
- 同一事实的数值不同（如"每天喂食2次" vs "每天喂食3次"）
- 安全性判断不同（如"可以少量喂食" vs "绝对不能喂"）
- 治疗方案矛盾
- 症状描述明显不同
- 品种易感性说法不一致

冲突严重度：
- **高**: 涉及安全/健康，可能导致错误决策（如食物安全性矛盾、用药剂量冲突）
- **中**: 数值或程度描述不一致，但不影响安全
- **低**: 措辞或表述差异，本质含义相近
"""

SINGLE_ARTICLE_PROMPT = """你是一个专业的宠物医疗知识编辑，负责将单篇原始文章转换为结构化的中文 Wiki 条目。

## 核心原则

1. **零信息损失**: 原文中的每一条事实、数据、建议都必须出现在 Wiki 中
2. **忠实翻译**: 英文内容翻译为中文，保留关键英文术语（括号标注）
3. **结构化**: 按照合适的 Wiki 模板组织信息

## 输出格式

请严格按以下 JSON 格式输出：

{
  "wiki_title": "中文 Wiki 标题",
  "wiki_content": "完整的 Markdown 格式 Wiki 正文",
  "conflicts": [],
  "all_sources_used": ["来源名"],
  "information_coverage": "完整",
  "notes_for_reviewer": ""
}

Wiki 正文中请在每个主要章节末尾用 `[来源: xxx]` 标注信息出处。
"""


# ============================================================
# 工具函数
# ============================================================

def read_markdown_file(filepath: str) -> tuple:
    """读取 Markdown 文件，返回 (metadata, body)"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    match = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)', content, re.DOTALL)
    if match:
        fm_text = match.group(1)
        body = match.group(2)
        metadata = {}
        for line in fm_text.split('\n'):
            if ':' in line:
                key, _, value = line.partition(':')
                key = key.strip()
                value = value.strip().strip('"')
                if value.startswith('['):
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        pass
                metadata[key] = value
        return metadata, body
    return {}, content


TOPIC_SYNONYMS = {
    "cataracts": ["白内障", "cataract"],
    "pancreatitis": ["胰腺炎", "胰脏"],
    "bladder stones": ["膀胱结石", "urolithiasis", "尿路结石", "urinary stones"],
    "ear infection": ["耳部感染", "otitis", "外耳炎", "中耳炎", "耳道感染"],
    "skin allergy": ["皮肤过敏", "dermatitis", "特应性皮炎", "atopic", "allergic dermatitis"],
    "hypothyroidism": ["甲状腺功能减退", "甲减"],
    "patellar luxation": ["髌骨脱位", "膝关节脱位"],
    "liver shunt": ["肝分流", "portosystemic shunt"],
    "dental disease": ["牙齿疾病", "口腔护理", "periodontal"],
    "diabetes": ["糖尿病", "血糖"],
    "cushing": ["库欣综合征", "hyperadrenocorticism", "肾上腺皮质机能亢进"],
    "epilepsy": ["癫痫", "seizure", "抽搐"],
    "schnauzer comedo": ["粉刺综合征", "comedo syndrome"],
    "vaccination": ["疫苗接种", "vaccine schedule", "免疫"],
    "deworming": ["驱虫", "寄生虫"],
    "spay neuter": ["绝育", "绝育手术", "去势"],
    "miniature schnauzer": ["迷你雪纳瑞", "迷你型", "迷雪"],
    "standard schnauzer": ["标准雪纳瑞", "标准型", "标雪"],
    "giant schnauzer": ["巨型雪纳瑞", "巨型", "巨雪"],
    "hand stripping": ["手剥", "手拔", "剥毛"],
    "grooming": ["美容", "造型", "毛发护理"],
    "barking": ["吠叫", "叫声控制"],
    "socialization": ["社会化", "社交"],
    "separation anxiety": ["分离焦虑"],
    "obesity": ["肥胖", "减肥", "体重管理"],
}


def _build_synonym_lookup():
    """构建双向同义词查找表: 任何同义词 → canonical key"""
    lookup = {}
    for canonical, synonyms in TOPIC_SYNONYMS.items():
        canonical_lower = canonical.lower()
        lookup[canonical_lower] = canonical_lower
        for s in synonyms:
            lookup[s.lower()] = canonical_lower
    return lookup

_SYNONYM_LOOKUP = _build_synonym_lookup()


def _normalize_topic_key(title: str, category: str) -> str:
    """
    标准化主题 key 用于分组。

    1. 移除常见前后缀
    2. 尝试通过同义词表匹配到 canonical key
    3. 同一疾病的中英文文章会归到同一组
    """
    normalized = title.lower()
    for pattern in [' in dogs', ' for dogs', ' - dogs', '狗狗的', '犬的', '狗的',
                    '犬', '：', ':', '（', '）', '(', ')', '症状原因及治疗',
                    'signs, causes, and treatment', 'signs causes and treatment',
                    'everything pet parents should know', '的', ' dog']:
        normalized = normalized.replace(pattern, '')
    normalized = re.sub(r'\s+', ' ', normalized).strip()

    # 尝试匹配同义词表
    for term, canonical in _SYNONYM_LOOKUP.items():
        if term in normalized:
            return f"{category}::{canonical}"

    return f"{category}::{normalized}"


def group_articles_by_topic(knowledge_dir: str) -> dict:
    """
    将知识库中的文章按主题分组。

    使用同义词表做语义合并，确保同一主题的中英文文章归到同一组。
    """
    skip_dirs = {"模板", "00-索引"}
    groups = defaultdict(list)

    for root, dirs, files in os.walk(knowledge_dir):
        dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith('.')]

        for filename in files:
            if not filename.endswith('.md') or filename.startswith('_'):
                continue

            filepath = os.path.join(root, filename)
            metadata, body = read_markdown_file(filepath)

            if metadata.get("wiki_processed") == "true":
                continue

            if len(body.strip()) < 50:
                continue

            category = os.path.basename(root)
            title = metadata.get("title", filename.replace('.md', ''))

            group_key = _normalize_topic_key(title, category)

            groups[group_key].append({
                "filepath": filepath,
                "metadata": metadata,
                "body": body,
                "title": title,
            })

    return dict(groups)


def write_wiki_entry(wiki_dir: str, category: str, title: str, content: str, metadata: dict):
    """写入 Wiki 条目"""
    # 确定目标目录
    target_dir = os.path.join(wiki_dir, category)
    os.makedirs(target_dir, exist_ok=True)

    # 安全文件名
    safe_name = re.sub(r'[^\w\u4e00-\u9fff\s-]', '', title)
    safe_name = re.sub(r'\s+', '-', safe_name.strip())[:80]
    if not safe_name:
        safe_name = "untitled"
    filename = f"{safe_name}.md"

    filepath = os.path.join(target_dir, filename)

    # 构建 frontmatter
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("---\n")
        f.write(f"title: \"{title}\"\n")
        f.write(f"category: {category}\n")
        f.write(f"sources: {json.dumps(metadata.get('sources', []), ensure_ascii=False)}\n")
        f.write(f"source_files: {json.dumps(metadata.get('source_files', []), ensure_ascii=False)}\n")
        f.write(f"has_conflicts: {'true' if metadata.get('has_conflicts') else 'false'}\n")
        f.write(f"generated_date: {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write(f"model: {CHAT_MODEL}\n")
        f.write("---\n\n")
        f.write(content)

    logger.info(f"  Wiki 条目已写入: {filepath}")
    return filepath


def append_conflict_report(conflicts_dir: str, conflicts: list, wiki_title: str, sources: list):
    """追加冲突报告"""
    if not conflicts:
        return

    os.makedirs(conflicts_dir, exist_ok=True)
    report_path = os.path.join(conflicts_dir, "冲突报告.md")

    with open(report_path, 'a', encoding='utf-8') as f:
        f.write(f"\n## {wiki_title}\n\n")
        f.write(f"**涉及来源**: {', '.join(sources)}\n")
        f.write(f"**检测时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        for i, conflict in enumerate(conflicts, 1):
            severity_icon = {"高": "🔴", "中": "🟡", "低": "🟢"}.get(conflict.get("severity", "中"), "⚪")
            f.write(f"### {severity_icon} 冲突 {i}: {conflict.get('topic', '未知')}\n\n")
            f.write(f"**描述**: {conflict.get('description', '')}\n\n")

            src_a = conflict.get("source_a", {})
            src_b = conflict.get("source_b", {})
            f.write(f"| 来源 | 说法 |\n")
            f.write(f"|------|------|\n")
            f.write(f"| {src_a.get('source', '?')} | {src_a.get('claim', '?')} |\n")
            f.write(f"| {src_b.get('source', '?')} | {src_b.get('claim', '?')} |\n\n")

            f.write(f"**严重度**: {conflict.get('severity', '中')}\n\n")
            f.write(f"**待处理**: - [ ] 人工审核\n\n")
            f.write("---\n")


def mark_source_processed(filepath: str):
    """标记源文件已处理"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    content = content.replace('wiki_processed: false', 'wiki_processed: true', 1)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)


# ============================================================
# Wiki 生成核心
# ============================================================

def generate_wiki_for_group(llm, group_key: str, articles: list) -> dict:
    """
    为一组相关文章生成 Wiki 条目

    Args:
        llm: LLM 实例
        group_key: 分组 key (如 "03-健康医疗::pancreatitis")
        articles: 该组的所有文章

    Returns:
        dict: LLM 返回的结果
    """
    category = group_key.split("::")[0] if "::" in group_key else "综合"

    # 如果只有一篇文章，使用单篇模式
    if len(articles) == 1:
        return _generate_single_article_wiki(llm, articles[0], category)

    # 多篇文章合并模式
    return _generate_merged_wiki(llm, articles, category)


def _generate_single_article_wiki(llm, article: dict, category: str) -> dict:
    """单篇文章 → Wiki"""
    title = article["title"]
    source = article["metadata"].get("source", "未知")
    body = article["body"]

    # 截断过长内容
    if len(body) > 20000:
        body = body[:20000] + "\n\n[... 内容较长，已截断部分末尾 ...]"

    user_msg = f"""请将以下文章转换为结构化的中文 Wiki 条目。

**原始标题**: {title}
**来源**: {source}
**分类**: {category}

**完整原文内容**（必须全部保留到 Wiki 中）:

{body}
"""

    try:
        response = llm.invoke([
            SystemMessage(content=SINGLE_ARTICLE_PROMPT),
            HumanMessage(content=user_msg),
        ])

        result_text = response.content.strip()
        if result_text.startswith('```'):
            result_text = re.sub(r'^```\w*\n', '', result_text)
            result_text = re.sub(r'\n```$', '', result_text)

        return json.loads(result_text)

    except json.JSONDecodeError as e:
        logger.error(f"  JSON 解析失败: {title} - {e}")
        # 降级：直接包装原文
        return _fallback_wiki(article, category)
    except Exception as e:
        logger.error(f"  LLM 调用失败: {title} - {e}")
        return _fallback_wiki(article, category)


def _generate_merged_wiki(llm, articles: list, category: str) -> dict:
    """多篇文章合并 → Wiki"""
    # 构建所有文章的摘要
    articles_text_parts = []
    total_length = 0
    max_total = 30000  # 总字符限制

    for i, article in enumerate(articles, 1):
        source = article["metadata"].get("source", "未知")
        title = article["title"]
        body = article["body"]

        # 分配额度
        per_article_limit = max(3000, (max_total - total_length) // (len(articles) - i + 1))
        if len(body) > per_article_limit:
            body = body[:per_article_limit] + f"\n\n[... 来源 {source} 的内容较长，已截断部分末尾以适应处理限制 ...]"

        articles_text_parts.append(
            f"### 来源 {i}: {title} ({source})\n\n{body}"
        )
        total_length += len(articles_text_parts[-1])

    all_articles_text = "\n\n" + "=" * 60 + "\n\n".join(articles_text_parts)

    sources_list = [a["metadata"].get("source", "未知") for a in articles]

    user_msg = f"""请将以下 {len(articles)} 篇来自不同来源的相关文章合并为一个无损的中文 Wiki 条目。

**涉及来源**: {', '.join(set(sources_list))}
**分类**: {category}

重要要求：
- 每篇文章的所有信息都必须保留
- 信息有重叠的部分合并，但要注明多个来源都提到了
- 信息有冲突的部分，全部保留并在 conflicts 数组中详细记录
- 某个来源独有的信息，标注来源后保留

{all_articles_text}
"""

    try:
        response = llm.invoke([
            SystemMessage(content=WIKI_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ])

        result_text = response.content.strip()
        if result_text.startswith('```'):
            result_text = re.sub(r'^```\w*\n', '', result_text)
            result_text = re.sub(r'\n```$', '', result_text)

        return json.loads(result_text)

    except json.JSONDecodeError as e:
        logger.error(f"  JSON 解析失败（合并模式）- {e}")
        # 降级：逐篇处理
        logger.info("  降级为逐篇处理模式...")
        results = []
        for article in articles:
            r = _generate_single_article_wiki(llm, article, category)
            results.append(r)

        # 合并降级结果
        combined_content = ""
        all_conflicts = []
        for r in results:
            combined_content += r.get("wiki_content", "") + "\n\n---\n\n"
            all_conflicts.extend(r.get("conflicts", []))

        return {
            "wiki_title": results[0].get("wiki_title", "合并条目") if results else "合并条目",
            "wiki_content": combined_content,
            "conflicts": all_conflicts,
            "all_sources_used": sources_list,
            "information_coverage": "部分缺失（降级处理）",
            "notes_for_reviewer": "LLM 合并失败，已降级为逐篇处理后拼接，建议人工整合",
        }

    except Exception as e:
        logger.error(f"  LLM 调用失败（合并模式）- {e}")
        return _fallback_wiki(articles[0], category)


def _fallback_wiki(article: dict, category: str) -> dict:
    """降级方案：直接包装原文为 Wiki 格式"""
    title = article["title"]
    source = article["metadata"].get("source", "未知")
    body = article["body"]

    content = f"# {title}\n\n"
    content += f"> ⚠️ 此条目由降级模式生成（LLM 处理失败），为原文直接包装，待人工处理。\n\n"
    content += f"**来源**: {source}\n\n"
    content += body
    content += f"\n\n---\n\n## 来源与参考\n\n- {source}: {article['metadata'].get('url', 'N/A')}\n"

    return {
        "wiki_title": title,
        "wiki_content": content,
        "conflicts": [],
        "all_sources_used": [source],
        "information_coverage": "完整（原文保留）",
        "notes_for_reviewer": "LLM 处理失败，此为原文直接包装，需要人工翻译/结构化",
    }


# ============================================================
# 主流程
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="LLM Wiki 生成器")
    parser.add_argument("--batch", type=int, default=0, help="每批处理的主题组数量（0=全部）")
    parser.add_argument("--category", type=str, help="只处理某分类（如 03-健康医疗）")
    parser.add_argument("--dry-run", action="store_true", help="仅列出分组，不生成")
    args = parser.parse_args()

    logger.info("===== LLM Wiki 生成器启动 =====")
    logger.info(f"知识库: {KNOWLEDGE_DIR}")
    logger.info(f"Wiki 输出: {WIKI_DIR}")
    logger.info(f"冲突报告: {CONFLICTS_DIR}")
    logger.info(f"模型: {CHAT_MODEL}")

    # 初始化冲突报告
    os.makedirs(CONFLICTS_DIR, exist_ok=True)
    report_path = os.path.join(CONFLICTS_DIR, "冲突报告.md")
    if not os.path.exists(report_path):
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# 冲突报告\n\n")
            f.write(f"自动检测到的多源信息冲突，需要人工审核。\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            f.write("---\n\n")

    # 初始化 LLM
    try:
        from agent.llm import create_llm
    except ImportError:
        from llm import create_llm
    llm = create_llm(temperature=0.1, max_tokens=4096)

    # 1. 按主题分组
    logger.info("\n--- 阶段 1: 文章分组 ---")
    groups = group_articles_by_topic(KNOWLEDGE_DIR)

    # 过滤分类
    if args.category:
        groups = {k: v for k, v in groups.items() if k.startswith(args.category)}
        logger.info(f"过滤分类 '{args.category}': {len(groups)} 个主题组")

    logger.info(f"共 {sum(len(v) for v in groups.values())} 篇文章，分为 {len(groups)} 个主题组")

    for key, articles in groups.items():
        sources = set(a["metadata"].get("source", "?") for a in articles)
        logger.info(f"  [{key}] {len(articles)} 篇 (来源: {', '.join(sources)})")

    if not groups:
        logger.warning("没有找到待处理的文章。请先运行爬虫和数据清洗。")
        return

    if args.dry_run:
        logger.info(f"\n--dry-run 模式，不生成文件。共 {len(groups)} 个主题组。")
        return

    # 2. 逐组生成 Wiki
    logger.info("\n--- 阶段 2: 生成 Wiki 条目 ---")
    stats = {"total": 0, "success": 0, "conflicts_found": 0, "fallback": 0}

    group_items = list(groups.items())
    if args.batch > 0:
        group_items = group_items[:args.batch]
        logger.info(f"批次模式: 只处理前 {args.batch} 个主题组")

    for group_key, articles in group_items:
        stats["total"] += 1
        category = group_key.split("::")[0] if "::" in group_key else "综合"

        logger.info(f"\n处理: {group_key} ({len(articles)} 篇)")
        result = generate_wiki_for_group(llm, group_key, articles)

        if not result:
            continue

        wiki_title = result.get("wiki_title", "未命名")
        wiki_content = result.get("wiki_content", "")
        conflicts = result.get("conflicts", [])
        sources_used = result.get("all_sources_used", [])
        notes = result.get("notes_for_reviewer", "")

        # 写入 Wiki 条目
        if wiki_content:
            wiki_metadata = {
                "sources": sources_used,
                "source_files": [a["filepath"] for a in articles],
                "has_conflicts": len(conflicts) > 0,
            }
            write_wiki_entry(WIKI_DIR, category, wiki_title, wiki_content, wiki_metadata)
            stats["success"] += 1

        # 记录冲突
        if conflicts:
            stats["conflicts_found"] += len(conflicts)
            append_conflict_report(CONFLICTS_DIR, conflicts, wiki_title, sources_used)
            logger.warning(f"  ⚠️ 检测到 {len(conflicts)} 个冲突")

        # 标注降级
        if "降级" in result.get("information_coverage", "") or "失败" in result.get("notes_for_reviewer", ""):
            stats["fallback"] += 1

        # 审核备注
        if notes:
            logger.info(f"  📝 审核备注: {notes}")

        # 标记源文件已处理
        for article in articles:
            mark_source_processed(article["filepath"])

    # 3. 汇总
    logger.info(f"\n===== Wiki 生成完成 =====")
    logger.info(f"总主题组: {stats['total']}")
    logger.info(f"成功生成: {stats['success']}")
    logger.info(f"检测到冲突: {stats['conflicts_found']} 个")
    logger.info(f"降级处理: {stats['fallback']} 个")
    logger.info(f"\nWiki 目录: {WIKI_DIR}")
    logger.info(f"冲突报告: {os.path.join(CONFLICTS_DIR, '冲突报告.md')}")

    if stats["conflicts_found"] > 0:
        logger.warning(f"\n⚠️  有 {stats['conflicts_found']} 个冲突需要人工审核！")
        logger.warning(f"请查看: {os.path.join(CONFLICTS_DIR, '冲突报告.md')}")

    # 4. 自动生成检索索引
    logger.info("\n--- 阶段 4: 生成检索索引 ---")
    try:
        try:
            from agent.build_index import main as build_index_main
        except ImportError:
            from build_index import main as build_index_main
        build_index_main()
    except ImportError:
        logger.warning("无法导入 build_index，跳过索引生成")
    except Exception as e:
        logger.warning(f"索引生成失败: {e}")

    # 5. 自动同步到 Obsidian
    if os.environ.get("OBSIDIAN_VAULT_DIR"):
        logger.info("\n--- 阶段 5: 同步到 Obsidian ---")
        try:
            from agent.sync_to_obsidian import main as sync_main
            sync_main()
        except ImportError:
            # 直接运行时的路径兼容
            try:
                from sync_to_obsidian import main as sync_main
                sync_main()
            except ImportError:
                logger.warning("无法导入 sync_to_obsidian，跳过 Obsidian 同步")
    else:
        logger.info("\n未配置 OBSIDIAN_VAULT_DIR，跳过 Obsidian 同步")


if __name__ == "__main__":
    main()
