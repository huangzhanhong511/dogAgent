"""
波奇网爬虫
爬取雪纳瑞相关中文养护知识
"""

import re
from urllib.parse import urljoin, urlparse, quote
from base_crawler import BaseCrawler


class BoqiiCrawler(BaseCrawler):
    """波奇网爬虫"""

    def __init__(self):
        super().__init__("boqii")
        self.base_url = "https://www.boqii.com"
        self.search_keywords = self.source_config.get("search_keywords", [])

        self.direct_urls = [
            "https://www.boqii.com/breed/detail/50.html",  # 雪纳瑞品种页
        ]

    def _is_relevant_url(self, url: str) -> bool:
        parsed = urlparse(url)
        if 'boqii.com' not in parsed.netloc:
            return False
        path = parsed.path.lower()
        exclude = ['.jpg', '.png', '.gif', '/user/', '/shop/', '/mall/']
        return not any(e in path for e in exclude)

    def extract_article_content(self, soup, url=""):
        article = {
            "title": "",
            "content_html": "",
            "content_markdown": "",
            "author": "波奇网",
            "date": "",
            "tags": [],
            "metadata": {"category": "综合"}
        }

        title_tag = soup.find('h1')
        if title_tag:
            article["title"] = title_tag.get_text(strip=True)

        content_selectors = [
            '.article-content',
            '.detail-content',
            '.content',
            'article',
            '.main-content',
        ]

        content_elem = None
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem and len(content_elem.get_text(strip=True)) > 50:
                break

        if content_elem:
            for unwanted in content_elem.select('script, style, .ad, .sidebar, nav'):
                unwanted.decompose()
            article["content_html"] = str(content_elem)
            article["content_markdown"] = self.html_to_markdown(str(content_elem))

        # 自动分类
        text = (article["title"] + article.get("content_markdown", "")).lower()
        if any(kw in text for kw in ['疾病', '生病', '治疗', '症状', '医院']):
            article["metadata"]["category"] = "健康医疗"
        elif any(kw in text for kw in ['饮食', '狗粮', '喂养', '食物', '营养']):
            article["metadata"]["category"] = "饮食营养"
        elif any(kw in text for kw in ['美容', '修剪', '洗澡', '造型', '梳毛']):
            article["metadata"]["category"] = "美容护理"
        elif any(kw in text for kw in ['训练', '训犬', '行为', '社会化']):
            article["metadata"]["category"] = "训练与行为"

        return article

    def _crawl_article(self, url: str):
        if self.is_crawled(url):
            self.stats["skipped"] += 1
            return

        html = self.fetch_page(url)
        if not html:
            return

        self.save_raw_html(url, html)
        soup = self.parse_html(html)
        article = self.extract_article_content(soup, url)

        if article["title"] and article.get("content_markdown") and len(article["content_markdown"]) > 100:
            self.save_article(url, article)

    def _crawl_search(self):
        for keyword in self.search_keywords:
            self.logger.info(f"搜索: {keyword}")
            search_url = f"{self.base_url}/search/?q={quote(keyword)}"

            html = self.fetch_page(search_url)
            if not html:
                continue

            soup = self.parse_html(html)
            links = self.get_links_from_page(
                soup, self.base_url,
                filter_fn=self._is_relevant_url
            )

            self.logger.info(f"  搜索 '{keyword}' 找到 {len(links)} 个链接")
            for link in links[:15]:
                self._crawl_article(link)

    def crawl(self):
        self.logger.info("--- 阶段 1: 爬取直接目标页面 ---")
        for url in self.direct_urls:
            self._crawl_article(url)

        self.logger.info("--- 阶段 2: 搜索补充内容 ---")
        self._crawl_search()


if __name__ == "__main__":
    crawler = BoqiiCrawler()
    crawler.run()