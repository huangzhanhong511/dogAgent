"""
统一运行所有爬虫的入口脚本
用法:
    python run_all.py              # 运行所有爬虫
    python run_all.py akc petmd    # 只运行指定爬虫
    python run_all.py --list       # 列出所有可用爬虫
"""

import sys
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("run_all")

CRAWLERS = {
    "akc": ("crawl_akc", "AKCCrawler"),
    "petmd": ("crawl_petmd", "PetMDCrawler"),
    "aspca": ("crawl_aspca", "ASPCACrawler"),
    "vca": ("crawl_vca", "VCACrawler"),
    "boqii": ("crawl_boqii", "BoqiiCrawler"),
    "reddit": ("crawl_reddit", "RedditCrawler"),
}


def run_crawler(name: str):
    """运行单个爬虫"""
    if name not in CRAWLERS:
        logger.error(f"未知爬虫: {name}")
        return False

    module_name, class_name = CRAWLERS[name]
    try:
        module = __import__(module_name)
        crawler_class = getattr(module, class_name)
        crawler = crawler_class()
        crawler.run()
        return True
    except Exception as e:
        logger.error(f"运行爬虫 {name} 失败: {e}", exc_info=True)
        return False


def main():
    args = sys.argv[1:]

    if "--list" in args:
        print("可用爬虫:")
        for name in CRAWLERS:
            print(f"  - {name}")
        return

    targets = args if args else list(CRAWLERS.keys())
    start = datetime.now()
    results = {}

    logger.info(f"===== 开始批量爬取 ({len(targets)} 个来源) =====")
    for name in targets:
        logger.info(f"\n>>> 运行: {name}")
        results[name] = run_crawler(name)

    elapsed = datetime.now() - start
    logger.info(f"\n===== 批量爬取完成 =====")
    logger.info(f"耗时: {elapsed}")
    for name, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        logger.info(f"  {name}: {status}")


if __name__ == "__main__":
    main()