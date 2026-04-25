"""
LLM 知识生成器 — 为知识库中缺失的主题生成内容

读取 target_topics.json 中的目标主题清单，扫描 knowledge/ 中已有内容，
对缺失的主题调用 LLM 生成中文文章，写入 knowledge/{category}/。

生成的文章标注 source: LLM-generated + reliability: 低 + needs_review: true，
后续由 build_wiki.py 统一处理成 wiki。

用法:
    python agent/generate_topics.py                    # 生成所有缺失主题
    python agent/generate_topics.py --dry-run           # 仅列出缺失主题
    python agent/generate_topics.py --category 03-健康医疗  # 只生成某分类
    python agent/generate_topics.py --batch-size 5      # 每批5个（默认10）
"""

import os
import re
import sys
import json
import time
import logging
import argparse
from datetime import datetime

from dotenv import load_dotenv

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
load_dotenv(os.path.join(PROJECT_DIR, ".env"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("generate_topics")

KNOWLEDGE_DIR = os.path.join(PROJECT_DIR, "knowledge")
TEMPLATE_DIR = os.path.join(KNOWLEDGE_DIR, "模板")
TOPICS_FILE = os.path.join(PROJECT_DIR, "crawlers", "config", "target_topics.json")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "qwen-plus")

CATEGORY_DIR_MAP = {
    "01-品种百科": "01-品种百科",
    "02-饮食营养": "02-饮食营养",
    "03-健康医疗": "03-健康医疗",
    "04-美容护理": "04-美容护理",
    "05-训练与行为": "05-训练与行为",
    "06-日常饲养": "06-日常饲养",
    "07-繁殖与幼犬": "07-繁殖与幼犬",
    "08-法规与养犬常识": "08-法规与养犬常识",
}

GENERATE_SYSTEM_PROMPT = """你是一位专业的宠物医疗与养护知识编辑，专精于雪纳瑞犬（Miniature / Standard / Giant Schnauzer）。

你的任务是根据主题撰写一篇高质量的中文知识百科文章。

## 写作要求

1. **雪纳瑞聚焦**: 所有内容必须突出雪纳瑞的特殊性（易感性、品种特点、注意事项）
2. **专业准确**: 使用兽医学术语，保留关键英文术语（用括号标注），如"胰腺炎（Pancreatitis）"
3. **实用性强**: 包含具体的数据、剂量、时间表、费用参考
4. **结构清晰**: 使用 Markdown 格式，层次分明
5. **标注不确定信息**: 对于具体的数字（如药物剂量、费用），如果不完全确定请用 [需验证] 标注
6. **引用来源**: 在文末标注信息来源（如 PetMD、AKC、VCA 等权威机构）

## 写作长度

目标 1500-3000 字。健康医疗类文章偏长，日常饲养类偏短。
"""


def create_llm():
    try:
        from agent.llm import create_llm as _create
    except ImportError:
        from llm import create_llm as _create
    return _create(temperature=0.2, max_tokens=4096)


def load_target_topics():
    with open(TOPICS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["topics"]


def load_template(template_name):
    if not template_name:
        return None
    path = os.path.join(TEMPLATE_DIR, f"{template_name}.md")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    body_match = re.search(r"^---\s*\n.*?\n---\s*\n(.+)", content, re.DOTALL)
    return body_match.group(1).strip() if body_match else content


def scan_existing_knowledge():
    existing = set()
    for root, _, files in os.walk(KNOWLEDGE_DIR):
        for f in files:
            if f.endswith(".md") and "模板" not in root and "00-索引" not in root:
                name = os.path.splitext(f)[0].lower()
                existing.add(name)
                path = os.path.join(root, f)
                with open(path, "r", encoding="utf-8") as fh:
                    content = fh.read(500)
                title_match = re.search(r"title:\s*[\"']?(.+?)[\"']?\s*$", content, re.MULTILINE)
                if title_match:
                    existing.add(title_match.group(1).strip().lower())
    return existing


def _safe_filename(title):
    safe = re.sub(r'[^\w一-鿿\s-]', '', title)
    safe = re.sub(r'\s+', '-', safe.strip())
    return safe[:80] or "untitled"


def find_gaps(topics, existing):
    gaps = []
    for t in topics:
        title_zh = t["title_zh"].lower()
        title_en = t["title_en"].lower()
        tid = t["id"].lower()
        safe = _safe_filename(t["title_zh"]).lower()

        found = False
        for e in existing:
            if title_zh in e or title_en in e or tid in e or safe in e:
                found = True
                break
            if any(kw in e for kw in [title_zh, tid]):
                found = True
                break

        if not found:
            gaps.append(t)
    return gaps


def generate_article(llm, topic, template_text):
    from langchain_core.messages import HumanMessage, SystemMessage

    user_prompt = f"请为以下主题撰写一篇专业的中文知识百科文章：\n\n"
    user_prompt += f"**主题**: {topic['title_zh']}（{topic['title_en']}）\n"
    user_prompt += f"**分类**: {topic['category']}\n"
    user_prompt += f"**重点**: 突出雪纳瑞犬种的相关性\n"

    if template_text:
        user_prompt += f"\n请参考以下结构模板（可根据内容需要调整）：\n\n{template_text}\n"

    msgs = [
        SystemMessage(content=GENERATE_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    response = llm.invoke(msgs)
    return response.content


def save_article(topic, content):
    cat_dir = CATEGORY_DIR_MAP.get(topic["category"], topic["category"])
    out_dir = os.path.join(KNOWLEDGE_DIR, cat_dir)
    os.makedirs(out_dir, exist_ok=True)

    filename = _safe_filename(topic["title_zh"]) + ".md"
    path = os.path.join(out_dir, filename)

    now = datetime.now().strftime("%Y-%m-%d")
    frontmatter = f"""---
source: LLM-generated
url: ""
title: "{topic['title_zh']}（{topic['title_en']}）"
category: {topic['category']}
tags: []
language: zh
reliability: 低
author: ""
date: ""
crawl_date: ""
wiki_processed: false
generated_by: {CHAT_MODEL}
generated_date: {now}
needs_review: true
---

"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(frontmatter + content)

    logger.info(f"  已保存: {path} ({len(content)} 字)")
    return path


def main():
    parser = argparse.ArgumentParser(description="LLM 知识生成器")
    parser.add_argument("--dry-run", action="store_true", help="仅列出缺失主题")
    parser.add_argument("--category", type=str, help="只生成某分类")
    parser.add_argument("--batch-size", type=int, default=10, help="每批生成数量")
    parser.add_argument("--priority", type=str, help="只生成某优先级 (P0/P1/P2)")
    args = parser.parse_args()

    topics = load_target_topics()
    logger.info(f"目标主题清单: {len(topics)} 个")

    if args.category:
        topics = [t for t in topics if t["category"] == args.category]
        logger.info(f"过滤分类 '{args.category}': {len(topics)} 个")

    if args.priority:
        topics = [t for t in topics if t["priority"] == args.priority]
        logger.info(f"过滤优先级 '{args.priority}': {len(topics)} 个")

    existing = scan_existing_knowledge()
    logger.info(f"已有知识文件关键词: {len(existing)} 个")

    gaps = find_gaps(topics, existing)
    logger.info(f"缺失主题: {len(gaps)} 个")

    if not gaps:
        logger.info("所有主题已有对应知识文件，无需生成")
        return

    for i, t in enumerate(gaps):
        logger.info(f"  [{t['priority']}] {t['title_zh']} ({t['category']})")

    if args.dry_run:
        logger.info(f"\n--dry-run 模式，不生成文件。共 {len(gaps)} 个缺失主题。")
        return

    llm = create_llm()
    generated = 0
    failed = 0

    for i, topic in enumerate(gaps):
        if generated >= args.batch_size:
            logger.info(f"\n已达批次上限 ({args.batch_size})，停止。剩余 {len(gaps) - i} 个主题。")
            break

        logger.info(f"\n[{i+1}/{len(gaps)}] 生成: {topic['title_zh']}")

        template_text = load_template(topic.get("template")) if topic.get("template") else None

        try:
            content = generate_article(llm, topic, template_text)
            save_article(topic, content)
            generated += 1
            time.sleep(3)
        except Exception as e:
            logger.error(f"  生成失败: {e}")
            failed += 1

    logger.info(f"\n生成完成: 成功 {generated} 篇，失败 {failed} 篇")


if __name__ == "__main__":
    main()
