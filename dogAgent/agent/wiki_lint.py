"""
dogAgent Wiki 知识库维护工具（Lint）

按 Karpathy LLM Wiki 的 lint 理念，定期检查和维护知识库健康度。

操作：
  --report       审计报告（默认）：相关性、覆盖度、冲突、孤立页面
  --prune        清理低相关性文章（移到 _archive/，可恢复）
  --crossref     添加交叉引用（基于关键词重叠）
  --fix-degraded 修复降级文章（重新走 LLM 处理）
  --all          依次执行 prune + crossref

用法:
    python agent/wiki_lint.py                # 审计报告
    python agent/wiki_lint.py --prune        # 清理无关文章
    python agent/wiki_lint.py --crossref     # 添加交叉引用
    python agent/wiki_lint.py --all          # prune + crossref
"""

import os
import sys
import json
import shutil
import re
import logging
import argparse

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("wiki_lint")

WIKI_DIR = os.path.join(PROJECT_DIR, "wiki")
INDEX_PATH = os.path.join(WIKI_DIR, "index.json")
ARCHIVE_DIR = os.path.join(WIKI_DIR, "_archive")
TOPICS_FILE = os.path.join(PROJECT_DIR, "crawlers", "config", "target_topics.json")

SCHNAUZER_KEYWORDS = {
    "雪纳瑞", "schnauzer", "迷你雪纳瑞", "miniature schnauzer",
    "标准雪纳瑞", "standard schnauzer", "巨型雪纳瑞", "giant schnauzer",
    "迷雪", "标雪", "巨雪", "mini schnauzer",
}


def load_index():
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def is_schnauzer_relevant(title, entry):
    """判断文章是否和雪纳瑞相关"""
    text = (title + " " + " ".join(entry.get("keywords", []))).lower()
    return any(kw in text for kw in SCHNAUZER_KEYWORDS)


# ── report ──

def cmd_report():
    """审计报告"""
    idx = load_index()
    total = len(idx)

    relevant = {t: e for t, e in idx.items() if is_schnauzer_relevant(t, e)}
    not_relevant = {t: e for t, e in idx.items() if not is_schnauzer_relevant(t, e)}

    # 降级文章
    degraded = []
    for t, e in idx.items():
        path = os.path.join(WIKI_DIR, e["path"])
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                if "降级模式" in f.read(500):
                    degraded.append(t)

    # 冲突
    conflicts = [t for t, e in idx.items() if e.get("has_conflicts")]

    # 交叉引用
    has_related = sum(1 for e in idx.values() if e.get("related"))

    # 覆盖度
    coverage = _check_coverage(idx)

    print(f"\n{'='*60}")
    print(f"dogAgent Wiki 审计报告")
    print(f"{'='*60}")
    print(f"\n📊 总览")
    print(f"  文章总数:           {total}")
    print(f"  雪纳瑞相关:         {len(relevant)} ({len(relevant)/total*100:.0f}%)")
    print(f"  非雪纳瑞通用文章:    {len(not_relevant)} ({len(not_relevant)/total*100:.0f}%)")
    print(f"  有交叉引用:         {has_related} / {total}")
    print(f"  降级文章:           {len(degraded)}")
    print(f"  冲突文章:           {len(conflicts)}")

    print(f"\n📂 分类分布（雪纳瑞相关）")
    by_cat = {}
    for t, e in relevant.items():
        cat = e.get("category", "?")
        by_cat[cat] = by_cat.get(cat, 0) + 1
    for cat, cnt in sorted(by_cat.items()):
        print(f"  {cat}: {cnt}")

    print(f"\n📂 非相关文章分布")
    by_cat2 = {}
    for t, e in not_relevant.items():
        cat = e.get("category", "?")
        by_cat2[cat] = by_cat2.get(cat, 0) + 1
    for cat, cnt in sorted(by_cat2.items()):
        print(f"  {cat}: {cnt}")

    if degraded:
        print(f"\n⚠️  降级文章")
        for t in degraded:
            print(f"  - {t}")

    if conflicts:
        print(f"\n⚠️  冲突文章")
        for t in conflicts:
            print(f"  - {t}")

    print(f"\n📋 主题覆盖度（对比 target_topics.json）")
    print(f"  已覆盖: {coverage['covered']} / {coverage['total']}")
    if coverage["missing"]:
        print(f"  缺失主题:")
        for m in coverage["missing"]:
            print(f"    - {m}")

    print(f"\n💡 建议操作")
    if len(not_relevant) > 0:
        print(f"  1. 运行 --prune 清理 {len(not_relevant)} 篇无关文章")
    if has_related == 0:
        print(f"  2. 运行 --crossref 添加交叉引用")
    if degraded:
        print(f"  3. 运行 --fix-degraded 修复 {len(degraded)} 篇降级文章")
    print()


def _check_coverage(idx):
    """检查 target_topics.json 的覆盖度"""
    if not os.path.exists(TOPICS_FILE):
        return {"total": 0, "covered": 0, "missing": []}

    with open(TOPICS_FILE, "r", encoding="utf-8") as f:
        topics = json.load(f)["topics"]

    titles_lower = {t.lower() for t in idx.keys()}
    keywords_all = set()
    for e in idx.values():
        keywords_all.update(kw.lower() for kw in e.get("keywords", []))

    covered = 0
    missing = []
    for topic in topics:
        zh = topic["title_zh"].lower()
        en = topic["title_en"].lower()
        tid = topic["id"].lower()
        found = any(
            zh in t or en in t or tid in t
            for t in titles_lower
        ) or zh in keywords_all or en in keywords_all
        if found:
            covered += 1
        else:
            missing.append(f"{topic['title_zh']} ({topic['category']})")

    return {"total": len(topics), "covered": covered, "missing": missing}


# ── prune ──

def cmd_prune():
    """清理低相关性文章到 _archive/"""
    idx = load_index()
    os.makedirs(ARCHIVE_DIR, exist_ok=True)

    moved = 0
    for title, entry in idx.items():
        if is_schnauzer_relevant(title, entry):
            continue

        src = os.path.join(WIKI_DIR, entry["path"])
        if not os.path.exists(src):
            continue

        cat = entry.get("category", "other")
        dest_dir = os.path.join(ARCHIVE_DIR, cat)
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, os.path.basename(src))

        shutil.move(src, dest)
        moved += 1

    logger.info(f"已移动 {moved} 篇无关文章到 {ARCHIVE_DIR}")
    logger.info("请运行 python agent/build_index.py 重建索引")


# ── crossref ──

CROSSREF_PROMPT = """你是一个知识库编辑，负责为 Wiki 文章添加交叉引用。

以下是知识库中所有文章的标题和摘要。请为每篇文章选出 3-5 篇最相关的文章（内容有关联、互补或用户可能同时想看的）。

规则：
1. 只基于主题相关性，不要因为都含有"雪纳瑞"就认为相关
2. 相关性示例：胰腺炎 ↔ 胰腺炎饮食管理、耳部感染 ↔ 耳部清洁护理、幼犬疫苗 ↔ 幼犬发育里程碑
3. 不要自引用（不要把文章自己列为相关）
4. 返回 JSON 对象，key 是文章标题，value 是相关文章标题数组

---
文章列表：

{articles}

---
请返回 JSON（不要 markdown 代码块）："""


def cmd_crossref():
    """用 LLM 做语义匹配添加交叉引用"""
    idx = load_index()
    titles = sorted(idx.keys())

    # 构建文章摘要列表（供 LLM 参考）
    article_lines = []
    for title in titles:
        entry = idx[title]
        summary = entry.get("summary", "")[:60]
        cat = entry.get("category", "")
        article_lines.append(f"- [{cat}] {title}: {summary}")
    articles_text = "\n".join(article_lines)

    logger.info(f"用 LLM 为 {len(idx)} 篇文章生成交叉引用...")

    try:
        llm = _create_llm()
    except Exception as e:
        logger.error(f"LLM 初始化失败: {e}，回退到关键词匹配")
        _crossref_keyword_fallback(idx)
        return

    from langchain_core.messages import HumanMessage

    # 分批：每批 20 篇文章，LLM 为这 20 篇从全量列表中选相关文章
    BATCH_SIZE = 20
    all_crossrefs = {}
    valid_titles = set(titles)

    for batch_start in range(0, len(titles), BATCH_SIZE):
        batch_titles = titles[batch_start:batch_start + BATCH_SIZE]
        batch_list = "\n".join(f"- {t}" for t in batch_titles)

        prompt = CROSSREF_PROMPT.format(articles=articles_text) + f"\n\n只需要为以下文章生成交叉引用：\n{batch_list}"

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```\w*\n?", "", raw)
                raw = re.sub(r"\n?```$", "", raw)
            batch_refs = json.loads(raw)
            all_crossrefs.update(batch_refs)
            logger.info(f"  批次 {batch_start//BATCH_SIZE + 1}: {len(batch_refs)} 篇")
        except Exception as e:
            logger.warning(f"  批次 {batch_start//BATCH_SIZE + 1} 失败: {e}")
            continue

    updates = 0
    for title, related_list in all_crossrefs.items():
        if title not in idx:
            continue
        valid_related = [r for r in related_list if r in valid_titles and r != title][:5]
        if valid_related:
            idx[title]["related"] = valid_related
            updates += 1

    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(idx, f, ensure_ascii=False, indent=2)

    logger.info(f"LLM 语义匹配完成，已更新 {updates} 篇文章的交叉引用")


def _create_llm():
    """创建 LLM 实例"""
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_DIR, ".env"))
    try:
        from agent.llm import create_llm
    except ImportError:
        from llm import create_llm
    return create_llm()


def _crossref_keyword_fallback(idx):
    """关键词重叠回退（LLM 不可用时）"""
    titles = list(idx.keys())
    stopwords = {"雪纳瑞", "schnauzer", "miniature schnauzer", "犬", "dog", "dogs",
                 "petmd", "index", "akc", "vca", "american kennel club"}

    kw_sets = {}
    for t, e in idx.items():
        kws = set(kw.lower() for kw in e.get("keywords", []) if len(kw) >= 2)
        kws -= stopwords
        kw_sets[t] = kws

    updates = 0
    for t1 in titles:
        kw1 = kw_sets.get(t1, set())
        if not kw1:
            continue

        scored = []
        for t2 in titles:
            if t1 == t2:
                continue
            kw2 = kw_sets.get(t2, set())
            if not kw2:
                continue
            overlap = len(kw1 & kw2)
            if overlap > 0:
                score = overlap / min(len(kw1), len(kw2))
                scored.append((t2, score))

        scored.sort(key=lambda x: -x[1])
        related = [t for t, s in scored[:5] if s >= 0.1]
        if related:
            idx[t1]["related"] = related
            updates += 1

    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(idx, f, ensure_ascii=False, indent=2)

    logger.info(f"关键词匹配回退完成，已更新 {updates} 篇文章的交叉引用")


# ── fix-degraded ──

def cmd_fix_degraded():
    """修复降级文章"""
    idx = load_index()
    degraded = []

    for t, e in idx.items():
        path = os.path.join(WIKI_DIR, e["path"])
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                if "降级模式" in f.read(500):
                    degraded.append((t, e))

    if not degraded:
        logger.info("没有降级文章需要修复")
        return

    logger.info(f"发现 {len(degraded)} 篇降级文章，尝试重新处理...")

    for title, entry in degraded:
        path = os.path.join(WIKI_DIR, entry["path"])
        # 找到对应的 knowledge 源文件
        source_files = entry.get("source_files", [])
        if source_files:
            logger.info(f"  {title} — 源文件: {source_files}")
            # 重置 wiki_processed 标志让 build_wiki 重新处理
            for sf in source_files:
                if os.path.exists(sf):
                    with open(sf, "r", encoding="utf-8") as f:
                        content = f.read()
                    content = content.replace("wiki_processed: true", "wiki_processed: false")
                    with open(sf, "w", encoding="utf-8") as f:
                        f.write(content)
            # 删除旧 wiki 文件
            os.remove(path)
            logger.info(f"  已删除旧文件，源文件已重置。请运行 build_wiki.py 重新生成。")
        else:
            logger.warning(f"  {title} — 无源文件信息，需手动处理")


# ── main ──

def main():
    parser = argparse.ArgumentParser(description="Wiki 知识库维护工具")
    parser.add_argument("--report", action="store_true", help="审计报告（默认）")
    parser.add_argument("--prune", action="store_true", help="清理低相关性文章")
    parser.add_argument("--crossref", action="store_true", help="添加交叉引用")
    parser.add_argument("--fix-degraded", action="store_true", help="修复降级文章")
    parser.add_argument("--all", action="store_true", help="执行 prune + crossref")
    args = parser.parse_args()

    if args.all:
        cmd_prune()
        cmd_crossref()
    elif args.prune:
        cmd_prune()
    elif args.crossref:
        cmd_crossref()
    elif args.fix_degraded:
        cmd_fix_degraded()
    else:
        cmd_report()


if __name__ == "__main__":
    main()
