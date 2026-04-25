"""
Obsidian 同步脚本

将 wiki/ 目录下的 Wiki 文件同步到 Obsidian Vault，并增强 Obsidian 特性：
1. 添加 [[双向链接]]
2. 添加 #标签
3. 生成 MOC（Map of Content）索引页
4. 用 Obsidian callout 语法标注冲突

用法:
    python agent/sync_to_obsidian.py
"""

import os
import re
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("sync_obsidian")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIKI_DIR = os.path.join(PROJECT_DIR, "wiki")
CONFLICTS_DIR = os.path.join(PROJECT_DIR, "conflicts")
OBSIDIAN_VAULT = os.environ.get("OBSIDIAN_VAULT_DIR", "")
OBSIDIAN_SUBFOLDER = "dogAgent"  # Vault 内的子文件夹名

# 分类名 → 中文标签映射
CATEGORY_TAGS = {
    "01-品种百科": "#品种百科",
    "02-饮食营养": "#饮食营养",
    "03-健康医疗": "#健康医疗",
    "04-美容护理": "#美容护理",
    "05-训练与行为": "#训练与行为",
    "06-日常饲养": "#日常饲养",
    "07-繁殖与幼犬": "#繁殖与幼犬",
    "08-法规与常识": "#法规与常识",
}


def read_wiki_file(filepath: str) -> dict:
    """读取 Wiki 文件，解析 frontmatter 和正文"""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    match = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)', content, re.DOTALL)
    if match:
        fm_text = match.group(1)
        body = match.group(2)
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
        return {"metadata": metadata, "body": body, "raw": content}
    return {"metadata": {}, "body": content, "raw": content}


def collect_all_wiki_titles(wiki_dir: str) -> dict:
    """收集所有 Wiki 条目的标题，用于生成双向链接"""
    titles = {}  # title -> relative_path (不含 .md)
    for root, dirs, files in os.walk(wiki_dir):
        for filename in files:
            if not filename.endswith(".md"):
                continue
            filepath = os.path.join(root, filename)
            data = read_wiki_file(filepath)
            title = data["metadata"].get("title", filename.replace(".md", ""))
            # Obsidian 链接路径
            rel_dir = os.path.relpath(root, wiki_dir)
            if rel_dir == ".":
                link_name = title
            else:
                link_name = title
            titles[title] = {
                "filename": filename,
                "category": rel_dir,
                "filepath": filepath,
                "has_conflicts": data["metadata"].get("has_conflicts") == "true",
                "sources": data["metadata"].get("sources", []),
            }
    return titles


def add_wikilinks(text: str, all_titles: dict, current_title: str) -> str:
    """
    在文本中自动添加 [[双向链接]]

    扫描所有已知的 Wiki 标题，如果在文本中出现则替换为 [[标题]]
    每个标题只替换第一次出现
    """
    linked_titles = set()

    for title in sorted(all_titles.keys(), key=len, reverse=True):
        if title == current_title:
            continue
        if title in linked_titles:
            continue

        # 避免在已有的 [[]] 或 frontmatter 中替换
        # 使用负向断言确保不在 [[ ]] 内部
        pattern = re.compile(
            r'(?<!\[\[)' + re.escape(title) + r'(?!\]\])',
            re.IGNORECASE
        )

        if pattern.search(text):
            text = pattern.sub(f"[[{title}]]", text, count=1)
            linked_titles.add(title)

    return text


def enhance_for_obsidian(body: str, metadata: dict, all_titles: dict) -> str:
    """增强 Wiki 内容以适配 Obsidian"""
    title = metadata.get("title", "")
    category = metadata.get("category", "")

    # 1. 添加标签块
    tags = ["#雪纳瑞"]
    cat_tag = CATEGORY_TAGS.get(category)
    if cat_tag:
        tags.append(cat_tag)

    # 从标题提取额外标签
    if "白内障" in title or "Cataract" in title.lower():
        tags.append("#疾病/白内障")
    if "标准" in title or "Standard" in title:
        tags.append("#品种/标准雪纳瑞")
    if "迷你" in title or "Miniature" in title:
        tags.append("#品种/迷你雪纳瑞")

    tag_line = f"**标签**: {' '.join(tags)}\n\n"

    # 2. 添加双向链接
    body = add_wikilinks(body, all_titles, title)

    # 3. 将冲突标记转换为 Obsidian callout
    body = re.sub(
        r'> ⚠️(.*?)$',
        r'> [!warning] ⚠️\1',
        body,
        flags=re.MULTILINE
    )

    # 4. 添加来源信息为 callout
    sources = metadata.get("sources", [])
    if isinstance(sources, str):
        try:
            sources = json.loads(sources)
        except:
            sources = [sources]

    # 5. 组装最终内容
    enhanced = tag_line + body

    # 6. 底部添加元信息
    enhanced += "\n\n---\n"
    enhanced += f"> [!info] 元信息\n"
    enhanced += f"> - 生成日期: {metadata.get('generated_date', 'N/A')}\n"
    enhanced += f"> - 模型: {metadata.get('model', 'N/A')}\n"
    enhanced += f"> - 来源: {', '.join(sources) if sources else 'N/A'}\n"

    if metadata.get("has_conflicts") == "true":
        enhanced += f"> - ⚠️ 存在信息冲突，请查看 [[冲突报告]]\n"

    return enhanced


def generate_moc(all_titles: dict) -> str:
    """生成 MOC（Map of Content）索引页"""
    moc = "# 🐾 dogAgent 雪纳瑞知识库\n\n"
    moc += f"> 自动生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
    moc += "这是 dogAgent 自动整理的雪纳瑞犬知识库索引。"
    moc += "点击链接可跳转到对应的 Wiki 条目。\n\n"

    # 按分类分组
    by_category = defaultdict(list)
    for title, info in all_titles.items():
        by_category[info["category"]].append((title, info))

    category_names = {
        "01-品种百科": "🐕 品种百科",
        "02-饮食营养": "🍖 饮食营养",
        "03-健康医疗": "🏥 健康医疗",
        "04-美容护理": "✂️ 美容护理",
        "05-训练与行为": "🎓 训练与行为",
        "06-日常饲养": "🏠 日常饲养",
        "07-繁殖与幼犬": "🐶 繁殖与幼犬",
        "08-法规与常识": "📋 法规与常识",
    }

    for cat_key in sorted(by_category.keys()):
        cat_name = category_names.get(cat_key, cat_key)
        entries = by_category[cat_key]

        moc += f"## {cat_name}\n\n"

        for title, info in sorted(entries, key=lambda x: x[0]):
            conflict_mark = " ⚠️" if info["has_conflicts"] else ""
            sources_str = ", ".join(info["sources"]) if info["sources"] else ""
            moc += f"- [[{title}]]{conflict_mark}"
            if sources_str:
                moc += f" — 来源: {sources_str}"
            moc += "\n"

        moc += "\n"

    # 统计信息
    total = len(all_titles)
    conflicts = sum(1 for t in all_titles.values() if t["has_conflicts"])
    moc += "---\n\n"
    moc += f"📊 **统计**: 共 {total} 个条目"
    if conflicts:
        moc += f"，其中 {conflicts} 个存在信息冲突（见 [[冲突报告]]）"
    moc += "\n"

    return moc


def sync_conflicts(conflicts_dir: str, obsidian_target: str):
    """同步冲突报告到 Obsidian"""
    report_path = os.path.join(conflicts_dir, "冲突报告.md")
    if not os.path.exists(report_path):
        return

    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 增强 Obsidian callout
    content = re.sub(
        r'### 🔴',
        '### 🔴\n> [!danger] 高严重度冲突\n>',
        content
    )
    content = re.sub(
        r'### 🟡',
        '### 🟡\n> [!warning] 中严重度冲突\n>',
        content
    )

    target_path = os.path.join(obsidian_target, "冲突报告.md")
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"  冲突报告已同步: {target_path}")


def main():
    if not OBSIDIAN_VAULT:
        logger.error("未配置 OBSIDIAN_VAULT_DIR，请在 .env 中设置")
        return

    if not os.path.isdir(OBSIDIAN_VAULT):
        logger.error(f"Obsidian Vault 目录不存在: {OBSIDIAN_VAULT}")
        return

    logger.info("===== Obsidian 同步启动 =====")
    logger.info(f"Wiki 源: {WIKI_DIR}")
    logger.info(f"Obsidian Vault: {OBSIDIAN_VAULT}")

    # 目标目录
    obsidian_target = os.path.join(OBSIDIAN_VAULT, OBSIDIAN_SUBFOLDER)
    os.makedirs(obsidian_target, exist_ok=True)

    # 1. 收集所有 Wiki 标题
    all_titles = collect_all_wiki_titles(WIKI_DIR)
    logger.info(f"共发现 {len(all_titles)} 个 Wiki 条目")

    if not all_titles:
        logger.warning("没有 Wiki 条目可同步。请先运行 build_wiki.py")
        return

    # 2. 逐文件处理并同步
    synced = 0
    for title, info in all_titles.items():
        filepath = info["filepath"]
        data = read_wiki_file(filepath)

        # 增强 Obsidian 特性
        enhanced_body = enhance_for_obsidian(
            data["body"], data["metadata"], all_titles
        )

        # 目标路径
        target_dir = os.path.join(obsidian_target, info["category"])
        os.makedirs(target_dir, exist_ok=True)

        # 使用中文标题作为文件名
        safe_title = re.sub(r'[/\\:*?"<>|]', '', title)
        target_file = os.path.join(target_dir, f"{safe_title}.md")

        # 写入（不含 frontmatter，Obsidian 原生支持但我们用 properties 格式）
        with open(target_file, "w", encoding="utf-8") as f:
            # Obsidian properties (YAML frontmatter)
            f.write("---\n")
            f.write(f"title: {title}\n")
            f.write(f"category: {info['category']}\n")
            sources = info.get("sources", [])
            if isinstance(sources, str):
                try:
                    sources = json.loads(sources)
                except:
                    sources = [sources]
            f.write(f"sources:\n")
            for s in sources:
                f.write(f"  - {s}\n")
            f.write(f"has_conflicts: {info['has_conflicts']}\n")
            f.write(f"synced_date: {datetime.now().strftime('%Y-%m-%d')}\n")
            f.write(f"aliases:\n  - {title}\n")
            f.write("---\n\n")
            f.write(enhanced_body)

        logger.info(f"  ✅ {title} → {os.path.relpath(target_file, OBSIDIAN_VAULT)}")
        synced += 1

    # 3. 同步冲突报告
    sync_conflicts(CONFLICTS_DIR, obsidian_target)

    # 4. 生成 MOC 索引页
    moc_content = generate_moc(all_titles)
    moc_path = os.path.join(OBSIDIAN_VAULT, "dogAgent-总览.md")
    with open(moc_path, "w", encoding="utf-8") as f:
        f.write(moc_content)
    logger.info(f"  📋 MOC 索引: {moc_path}")

    # 5. 汇总
    logger.info(f"\n===== 同步完成 =====")
    logger.info(f"同步条目: {synced}")
    logger.info(f"Obsidian 目标: {obsidian_target}")
    logger.info(f"MOC 索引: {moc_path}")
    logger.info(f"\n打开 Obsidian 即可看到最新内容 🎉")


if __name__ == "__main__":
    main()