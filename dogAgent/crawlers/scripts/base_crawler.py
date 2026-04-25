"""
基础爬虫类 - 所有爬虫脚本的父类
提供通用的请求、解析、保存功能
"""

import os
import json
import time
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
import html2text

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class BaseCrawler:
    """基础爬虫类"""

    def __init__(self, source_name: str, config_path: str = None):
        self.source_name = source_name
        self.logger = logging.getLogger(f"crawler.{source_name}")

        # 加载配置
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "config", "sources.json"
            )
        with open(config_path, 'r', encoding='utf-8') as f:
            full_config = json.load(f)

        self.source_config = full_config["sources"].get(source_name, {})
        self.crawl_settings = full_config["crawl_settings"]

        # 路径设置
        self.base_dir = os.path.dirname(os.path.dirname(__file__))
        self.raw_data_dir = os.path.join(self.base_dir, "raw_data", source_name)
        self.cleaned_data_dir = os.path.join(self.base_dir, "cleaned_data", source_name)
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.cleaned_data_dir, exist_ok=True)

        # 请求会话
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": self.crawl_settings["user_agent"],
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
        })

        # html2text 转换器配置
        self.h2t = html2text.HTML2Text()
        self.h2t.ignore_links = False
        self.h2t.ignore_images = True
        self.h2t.ignore_emphasis = False
        self.h2t.body_width = 0  # 不自动换行
        self.h2t.unicode_snob = True

        # 已爬取的 URL 记录
        self.crawled_urls_file = os.path.join(self.raw_data_dir, "_crawled_urls.json")
        self.crawled_urls = self._load_crawled_urls()

        # 统计
        self.stats = {
            "total_requests": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "start_time": None,
        }

    def _load_crawled_urls(self) -> dict:
        """加载已爬取的 URL 记录"""
        if os.path.exists(self.crawled_urls_file):
            with open(self.crawled_urls_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_crawled_urls(self):
        """保存已爬取的 URL 记录"""
        with open(self.crawled_urls_file, 'w', encoding='utf-8') as f:
            json.dump(self.crawled_urls, f, ensure_ascii=False, indent=2)

    def _url_to_filename(self, url: str) -> str:
        """将 URL 转换为安全的文件名"""
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        parsed = urlparse(url)
        path_parts = parsed.path.strip('/').split('/')
        if path_parts and path_parts[-1]:
            name = path_parts[-1][:50]  # 截取最后一部分，最多50字符
        else:
            name = parsed.netloc
        # 清理文件名
        safe_name = "".join(c if c.isalnum() or c in '-_' else '_' for c in name)
        return f"{safe_name}_{url_hash}"

    def fetch_page(self, url: str, retry: int = None) -> Optional[str]:
        """
        获取页面 HTML 内容

        Args:
            url: 目标 URL
            retry: 重试次数，默认使用配置中的值

        Returns:
            HTML 内容字符串，失败返回 None
        """
        if retry is None:
            retry = self.crawl_settings["max_retries"]

        self.stats["total_requests"] += 1

        for attempt in range(retry + 1):
            try:
                self.logger.info(f"正在获取: {url} (尝试 {attempt + 1}/{retry + 1})")
                response = self.session.get(
                    url,
                    timeout=self.crawl_settings["timeout"]
                )
                response.raise_for_status()

                # 尝试检测编码
                if response.encoding is None or response.encoding == 'ISO-8859-1':
                    response.encoding = response.apparent_encoding

                self.stats["successful"] += 1

                # 请求间隔
                time.sleep(self.crawl_settings["delay_between_requests"])
                return response.text

            except requests.RequestException as e:
                self.logger.warning(f"请求失败 ({attempt + 1}/{retry + 1}): {url} - {e}")
                if attempt < retry:
                    wait_time = self.crawl_settings["delay_between_requests"] * (attempt + 1)
                    self.logger.info(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"最终失败: {url}")
                    self.stats["failed"] += 1
                    return None

    def parse_html(self, html: str) -> BeautifulSoup:
        """解析 HTML 为 BeautifulSoup 对象"""
        return BeautifulSoup(html, 'lxml')

    def html_to_markdown(self, html: str) -> str:
        """将 HTML 转换为 Markdown"""
        return self.h2t.handle(html)

    def extract_article_content(self, soup: BeautifulSoup) -> dict:
        """
        从页面中提取文章内容（子类应重写此方法）

        Returns:
            dict: {
                "title": str,
                "content_html": str,
                "content_text": str,
                "author": str,
                "date": str,
                "tags": list,
                "metadata": dict
            }
        """
        raise NotImplementedError("子类必须实现 extract_article_content 方法")

    def save_raw_html(self, url: str, html: str):
        """保存原始 HTML"""
        filename = self._url_to_filename(url)
        filepath = os.path.join(self.raw_data_dir, f"{filename}.html")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
        self.logger.debug(f"原始 HTML 已保存: {filepath}")
        return filepath

    def save_article(self, url: str, article: dict):
        """
        保存提取的文章为 Markdown 格式

        Args:
            url: 文章来源 URL
            article: extract_article_content 返回的字典
        """
        filename = self._url_to_filename(url)
        filepath = os.path.join(self.cleaned_data_dir, f"{filename}.md")

        # 构建 frontmatter
        frontmatter = {
            "source": self.source_config.get("name", self.source_name),
            "url": url,
            "crawl_date": datetime.now().strftime("%Y-%m-%d"),
            "language": self.source_config.get("language", "en"),
            "reliability": self.source_config.get("reliability", "中"),
            "title": article.get("title", ""),
            "author": article.get("author", ""),
            "date": article.get("date", ""),
            "tags": article.get("tags", []),
        }
        # 合并额外的 metadata
        if article.get("metadata"):
            frontmatter.update(article["metadata"])

        # 写入文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("---\n")
            for key, value in frontmatter.items():
                if isinstance(value, list):
                    f.write(f"{key}: {json.dumps(value, ensure_ascii=False)}\n")
                elif isinstance(value, str) and ('\n' in value or ':' in value):
                    f.write(f'{key}: "{value}"\n')
                else:
                    f.write(f"{key}: {value}\n")
            f.write("---\n\n")

            # 标题
            title = article.get("title", "Untitled")
            f.write(f"# {title}\n\n")

            # 正文
            content = article.get("content_markdown") or article.get("content_text", "")
            f.write(content)

            # 来源引用
            f.write(f"\n\n---\n\n## 参考来源\n\n")
            f.write(f"- [{self.source_config.get('name', self.source_name)}]({url})\n")
            f.write(f"- 爬取时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

        # 更新已爬取记录
        self.crawled_urls[url] = {
            "filename": filename,
            "title": article.get("title", ""),
            "crawl_date": datetime.now().isoformat(),
        }
        self._save_crawled_urls()

        self.logger.info(f"文章已保存: {filepath}")
        return filepath

    def is_crawled(self, url: str) -> bool:
        """检查 URL 是否已爬取"""
        return url in self.crawled_urls

    def get_links_from_page(self, soup: BeautifulSoup, base_url: str,
                            selector: str = "a", filter_fn=None) -> list:
        """
        从页面提取链接

        Args:
            soup: BeautifulSoup 对象
            base_url: 基础 URL，用于拼接相对链接
            selector: CSS 选择器
            filter_fn: 链接过滤函数

        Returns:
            链接列表
        """
        links = []
        for a_tag in soup.select(selector):
            href = a_tag.get('href', '')
            if not href or href.startswith('#') or href.startswith('javascript:'):
                continue
            full_url = urljoin(base_url, href)
            if filter_fn is None or filter_fn(full_url):
                links.append(full_url)
        return list(set(links))  # 去重

    def crawl(self):
        """主爬取流程（子类应重写此方法）"""
        raise NotImplementedError("子类必须实现 crawl 方法")

    def run(self):
        """运行爬虫"""
        self.stats["start_time"] = datetime.now()
        self.logger.info(f"===== 开始爬取: {self.source_name} =====")

        try:
            self.crawl()
        except KeyboardInterrupt:
            self.logger.warning("用户中断爬取")
        except Exception as e:
            self.logger.error(f"爬取异常: {e}", exc_info=True)
        finally:
            elapsed = datetime.now() - self.stats["start_time"]
            self.logger.info(f"===== 爬取完成: {self.source_name} =====")
            self.logger.info(f"总请求: {self.stats['total_requests']}")
            self.logger.info(f"成功: {self.stats['successful']}")
            self.logger.info(f"失败: {self.stats['failed']}")
            self.logger.info(f"跳过: {self.stats['skipped']}")
            self.logger.info(f"耗时: {elapsed}")