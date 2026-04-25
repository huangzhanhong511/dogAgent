"""
ASPCA 爬虫
主要爬取有毒食物/植物数据库、急救指南、养犬护理
ASPCA 是最权威的动物毒物控制信息来源
"""

import re
from urllib.parse import urljoin, urlparse
from base_crawler import BaseCrawler


class ASPCACrawler(BaseCrawler):
    """ASPCA 网站爬虫"""

    def __init__(self):
        super().__init__("aspca")
        self.base_url = "https://www.aspca.org"

        # 核心目标页面
        self.direct_urls = [
            # 有毒食物
            "https://www.aspca.org/pet-care/animal-poison-control/people-foods-avoid-feeding-your-pets",
            # 有毒植物列表
            "https://www.aspca.org/pet-care/animal-poison-control/toxic-and-non-toxic-plants",
            # 狗狗护理
            "https://www.aspca.org/pet-care/dog-care",
            "https://www.aspca.org/pet-care/dog-care/dog-nutrition-tips",
            "https://www.aspca.org/pet-care/dog-care/dog-grooming-tips",
            "https://www.aspca.org/pet-care/dog-care/general-dog-care",
            "https://www.aspca.org/pet-care/dog-care/common-dog-diseases",
            # 毒物控制
            "https://www.aspca.org/pet-care/animal-poison-control",
        ]

    def _is_relevant_url(self, url: str) -> bool:
        """判断 URL 是否相关"""
        parsed = urlparse(url)
        if 'aspca.org' not in parsed.netloc:
            return False

        path = parsed.path.lower()
        relevant_paths = [
            '/pet-care/dog-care',
            '/pet-care/animal-poison-control',
            '/pet-care/general-pet-care',
        ]
        return any(rp in path for rp in relevant_paths)

    def extract_article_content(self, soup, url=""):
        """提取 ASPCA 文章内容"""
        article = {
            "title": "",
            "content_html": "",
            "content_markdown": "",
            "author": "ASPCA",
            "date": "",
            "tags": [],
            "metadata": {}
        }

        # 标题
        title_tag = soup.find('h1')
        if title_tag:
            article["title"] = title_tag.get_text(strip=True)

        # 正文提取
        content_selectors = [
            '.field--name-body',
            '.node__content',
            'article .content',
            '.main-content',
            'article',
        ]

        content_elem = None
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem and len(content_elem.get_text(strip=True)) > 50:
                break

        if content_elem:
            for unwanted in content_elem.select(
                'script, style, nav, .ad, .social-share, '
                '.related, .sidebar, .newsletter, footer'
            ):
                unwanted.decompose()

            article["content_html"] = str(content_elem)
            article["content_markdown"] = self.html_to_markdown(str(content_elem))

        # 分类
        if 'poison-control' in url:
            article["metadata"]["category"] = "饮食营养/禁忌食物"
            article["tags"].extend(["有毒", "禁忌", "中毒"])
            if 'plants' in url:
                article["metadata"]["category"] = "日常饲养/安全"
                article["tags"].append("植物")
            elif 'people-foods' in url:
                article["tags"].append("人类食物")
        elif 'dog-care' in url:
            article["metadata"]["category"] = "日常饲养"
            if 'nutrition' in url:
                article["metadata"]["category"] = "饮食营养"
            elif 'grooming' in url:
                article["metadata"]["category"] = "美容护理"
            elif 'disease' in url:
                article["metadata"]["category"] = "健康医疗"
        else:
            article["metadata"]["category"] = "综合"

        return article

    def _extract_toxic_plants_list(self, soup):
        """特殊处理：提取有毒植物列表"""
        plants = []
        plant_items = soup.select('.view-content .views-row, .plant-list-item, table tr')

        for item in plant_items:
            plant = {}
            name_elem = item.select_one('a, .field--name-title, td:first-child')
            if name_elem:
                plant["name"] = name_elem.get_text(strip=True)

            toxicity_elem = item.select_one('.field--name-field-toxicity, td:nth-child(2)')
            if toxicity_elem:
                plant["toxicity"] = toxicity_elem.get_text(strip=True)

            symptoms_elem = item.select_one('.field--name-field-clinical-signs, td:nth-child(3)')
            if symptoms_elem:
                plant["symptoms"] = symptoms_elem.get_text(strip=True)

            if plant.get("name"):
                plants.append(plant)

        return plants

    def _crawl_article(self, url: str):
        """爬取单篇文章"""
        if self.is_crawled(url):
            self.logger.info(f"跳过已爬取: {url}")
            self.stats["skipped"] += 1
            return

        html = self.fetch_page(url)
        if not html:
            return

        self.save_raw_html(url, html)
        soup = self.parse_html(html)

        # 特殊处理有毒植物页面
        if 'toxic-and-non-toxic-plants' in url:
            plants = self._extract_toxic_plants_list(soup)
            if plants:
                self.logger.info(f"  提取到 {len(plants)} 种植物信息")

        article = self.extract_article_content(soup, url)

        if article["title"] and article.get("content_markdown"):
            if len(article["content_markdown"]) > 100:
                self.save_article(url, article)
        else:
            self.logger.warning(f"无法提取有效内容: {url}")

    def _discover_subpages(self, url: str):
        """发现子页面"""
        html = self.fetch_page(url)
        if not html:
            return []

        soup = self.parse_html(html)
        links = self.get_links_from_page(
            soup, self.base_url,
            filter_fn=self._is_relevant_url
        )
        return links

    def crawl(self):
        """主爬取流程"""
        # 1. 爬取直接目标页面
        self.logger.info("--- 阶段 1: 爬取核心页面 ---")
        for url in self.direct_urls:
            self._crawl_article(url)

        # 2. 发现并爬取更多子页面
        self.logger.info("--- 阶段 2: 发现子页面 ---")
        seed_urls = [
            "https://www.aspca.org/pet-care/dog-care",
            "https://www.aspca.org/pet-care/animal-poison-control",
        ]

        for seed_url in seed_urls:
            sub_links = self._discover_subpages(seed_url)
            self.logger.info(f"  从 {seed_url} 发现 {len(sub_links)} 个子页面")
            for link in sub_links:
                self._crawl_article(link)


if __name__ == "__main__":
    crawler = ASPCACrawler()
    crawler.run()