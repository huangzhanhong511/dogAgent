"""
数据清洗与知识库组织脚本（保守策略）

核心原则：零信息损失
- 只去掉确定的垃圾信息（script/style/nav/广告）
- 正文内容全部保留，宁可冗余也不误删
- 保持原文完整性，翻译和结构化交给后续的 LLM Wiki 生成步骤
"""

import os
import re
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("clean_and_organize")

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANED_DATA_DIR = os.path.join(BASE_DIR, "cleaned_data")
KNOWLEDGE_DIR = os.path.join(os.path.dirname(BASE_DIR), "knowledge")

# 知识库分类目录映射
CATEGORY_DIR_MAP = {
    "品种百科": "01-品种百科",
    "饮食营养": "02-饮食营养",
    "饮食营养/禁忌食物": "02-饮食营养",
    "健康医疗": "03-健康医疗",
    "健康医疗/常见疾病": "03-健康医疗",
    "健康医疗/预防保健": "03-健康医疗",
    "健康医疗/急救指南": "03-健康医疗",
    "健康医疗/老年犬护理": "03-健康医疗",
    "健康医疗/药物清单": "03-健康医疗",
    "美容护理": "04-美容护理",
    "训练与行为": "05-训练与行为",
    "日常饲养": "06-日常饲养",
    "日常饲养/安全": "06-日常饲养",
    "繁殖与幼犬": "07-繁殖与幼犬",
    "法规与常识": "08-法规与常识",
    "综合": "09-参考资料",
}


def minimal_clean(content: str) -> str:
    """
    最小化清洗 — 只去掉确定的垃圾，保留一切正文内容

    只移除：
    - 残留的 <script>, <style>, <iframe> 标签及内容
    - 多余的连续空行（压缩为最多3个）
    - 行尾空白
    不移除：
    - 任何正文文本
    - 链接、图片引用
    - 列表、表格等结构
    """
    # 移除残留的 HTML 脚本/样式标签
    content = re.sub(r'<(script|style|iframe|noscript)[^>]*>.*?</\1>', '', content, flags=re.DOTALL | re.IGNORECASE)
    # 移除残留的 HTML 注释
    content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)
    # 压缩多余空行（保留最多3个，给 Markdown 段落留空间）
    content = re.sub(r'\n{5,}', '\n\n\n\n', content)
    # 移除行尾空白
    content = re.sub(r'[ \t]+\n', '\n', content)

    return content.strip()


def extract_frontmatter(content: str) -> tuple:
    """提取 frontmatter 和正文"""
    match = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)', content, re.DOTALL)
    if match:
        fm_text = match.group(1)
        body = match.group(2)

        metadata = {}
        for line in fm_text.split('\n'):
            if ':' in line:
                key, _, value = line.partition(':')
                key = key.strip()
                value = value.strip()
                if value.startswith('[') or value.startswith('{'):
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        pass
                elif value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                metadata[key] = value

        return metadata, body
    return {}, content


def generate_safe_filename(title: str, max_length: int = 80) -> str:
    """生成安全的文件名"""
    safe = re.sub(r'[^\w\u4e00-\u9fff\s-]', '', title)
    safe = re.sub(r'\s+', '-', safe.strip())
    if len(safe) > max_length:
        safe = safe[:max_length]
    return safe or "untitled"


def organize_file(filepath: str, source_name: str):
    """
    将爬取的文件组织到知识库中

    保守策略：
    - 正文只做最小化清洗
    - 保留所有原始内容
    - 只在 frontmatter 中添加/标准化元数据
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    frontmatter, body = extract_frontmatter(content)

    # 确定分类目录
    category = frontmatter.get("category", "综合")
    if isinstance(category, str):
        target_dir_name = CATEGORY_DIR_MAP.get(category, "09-参考资料")
    else:
        target_dir_name = "09-参考资料"

    target_dir = os.path.join(KNOWLEDGE_DIR, target_dir_name)
    os.makedirs(target_dir, exist_ok=True)

    # 生成文件名
    title = frontmatter.get("title", "")
    if not title:
        title = os.path.splitext(os.path.basename(filepath))[0]
    filename = generate_safe_filename(title) + ".md"

    # 最小化清洗正文（只去垃圾，不丢内容）
    cleaned_body = minimal_clean(body)

    # 标准化 frontmatter
    new_frontmatter = {
        "source": frontmatter.get("source", source_name),
        "url": frontmatter.get("url", ""),
        "title": title,
        "category": category,
        "tags": frontmatter.get("tags", []),
        "language": frontmatter.get("language", "en"),
        "reliability": frontmatter.get("reliability", "中"),
        "author": frontmatter.get("author", ""),
        "date": frontmatter.get("date", ""),
        "crawl_date": frontmatter.get("crawl_date", ""),
        "wiki_processed": "false",  # 标记是否已被 Wiki 生成器处理
    }

    # 写入目标文件
    target_path = os.path.join(target_dir, filename)

    # 避免覆盖
    if os.path.exists(target_path):
        base, ext = os.path.splitext(filename)
        target_path = os.path.join(target_dir, f"{base}_{source_name}{ext}")

    with open(target_path, 'w', encoding='utf-8') as f:
        f.write("---\n")
        for key, value in new_frontmatter.items():
            if isinstance(value, list):
                f.write(f"{key}: {json.dumps(value, ensure_ascii=False)}\n")
            elif isinstance(value, str) and (':' in value or '\n' in value):
                f.write(f'{key}: "{value}"\n')
            else:
                f.write(f"{key}: {value}\n")
        f.write("---\n\n")
        f.write(cleaned_body)

    logger.info(f"  -> {target_path}")
    return target_path


def process_source(source_name: str):
    """处理单个来源"""
    source_dir = os.path.join(CLEANED_DATA_DIR, source_name)
    if not os.path.exists(source_dir):
        logger.warning(f"来源目录不存在: {source_dir}")
        return 0

    count = 0
    for filename in os.listdir(source_dir):
        if not filename.endswith('.md') or filename.startswith('_'):
            continue

        filepath = os.path.join(source_dir, filename)
        try:
            organize_file(filepath, source_name)
            count += 1
        except Exception as e:
            logger.error(f"处理失败 {filepath}: {e}")

    return count


def main():
    logger.info("===== 开始数据清洗与组织（保守模式：零信息损失）=====")

    # 确保知识库目录存在
    for dir_name in set(CATEGORY_DIR_MAP.values()):
        os.makedirs(os.path.join(KNOWLEDGE_DIR, dir_name), exist_ok=True)

    total = 0
    sources = ["akc", "petmd", "aspca", "vca", "boqii", "reddit"]

    for source in sources:
        logger.info(f"\n处理来源: {source}")
        count = process_source(source)
        logger.info(f"  处理了 {count} 篇文章")
        total += count

    logger.info(f"\n===== 清洗完成，共处理 {total} 篇文章（正文内容全部保留）=====")


if __name__ == "__main__":
    main()