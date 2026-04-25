"""
VCA Animal Hospitals 爬虫
爬取兽医专业文章，内容权威性非常高
"""

import re
from urllib.parse import urljoin, urlparse, quote_plus
from base_crawler import BaseCrawler


class VCACrawler(BaseCrawler):
    """VCA 网站爬虫"""

    def __init__(self):
        super().__init__("vca")
        self.base_url = "https://vcahospitals.com"
        self.search_keywords = self.source_config.get("search_keywords", [])

        # 直接目标文章 - 雪纳瑞高发疾病的 VCA 文章
        self.direct_urls = [
            "https://vcahospitals.com/know-your-pet/pancreatitis-in-dogs",
            "https://vcahospitals.com/know-your-pet/bladder-stones-in-dogs",
            "https://vcahospitals.com/know-your-pet/struvite-bladder-stones-in-dogs",
            "https://vcahospitals.com/know-your-pet/calcium-oxalate-bladder-stones-in-dogs",
            "https://vcahospitals.com/know-your-pet/allergies-in-dogs",
            "https://vcahospitals.com/know-your-pet/ear-infections-otitis-externa-in-dogs",
            "https://vcahospitals.com/know-your-pet/cataracts-in-dogs",
            "https://vcahospitals.com/know-your-pet/luxating-patella-in-dogs",
            "https://vcahospitals.com/know-your-pet/hypothyroidism-in-dogs",
            "https://vcahospitals.com/know-your-pet/diabetes-mellitus-in-dogs-overview",
            "https://vcahospitals.com/know-your-pet/cushings-disease-in-dogs",
            "https://vcahospitals.com/know-your-pet/dental-disease-in-dogs",
            "https://vcahospitals.com/know-your-pet/seizures-in-dogs",
            "https://vcahospitals.com/know-your-pet/liver-disease-in-dogs",
            # 预防保健
            "https://vcahospitals.com/know-your-pet/vaccinations-for-dogs",
            "https://vcahospitals.com/know-your-pet/deworming-your-dog",
            "https://vcahospitals.com/know-your-pet/spaying-in-dogs",
            "https://vcahospitals.com/know-your-pet/neutering-in-dogs",
            # 营养
            "https://vcahospitals.com/know-your-pet/nutrition-general-feeding-guidelines-for-dogs",
            "https://vcahospitals.com/know-your-pet/feeding-your-puppy",
            "https://vcahospitals.com/know-your-pet/nutrition-for-senior-dogs",
            # 护理
            "https://vcahospitals.com/know-your-pet/grooming-and-coat-care-for-your-dog",
            "https://vcahospitals.com/know-your-pet/dental-care-for-dogs",
            "https://vcahospitals.com/know-your-pet/ear-cleaning-in-dogs",
        ]

    def _is_relevant_url(self, url: str) -> bool:
        parsed = urlparse(url)
        if 'vcahospitals.com' not in parsed.netloc:
            return False
        return '/know-your-pet/' in parsed.path

    def extract_article_content(self, soup, url=""):
        article = {
            "title": "",
            "content_html": "",
            "content_markdown": "",
            "author": "VCA Animal Hospitals",
            "date": "",
            "tags": [],
            "metadata": {}
        }

        title_tag = soup.find('h1')
        if title_tag:
            article["title"] = title_tag.get_text(strip=True)

        # VCA 文章通常标注审核兽医
        reviewer = soup.find(class_=re.compile(r'author|reviewer|veterinarian', re.I))
        if reviewer:
            article["author"] = f"VCA - {reviewer.get_text(strip=True)}"

        content_selectors = [
            '.article-content',
            '.field--name-body',
            '.node__content',
            'article .content',
            '.main-content article',
            'article',
        ]

        content_elem = None
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem and len(content_elem.get_text(strip=True)) > 100:
                break

        if content_elem:
            for unwanted in content_elem.select(
                'script, style, nav, .ad, .social, .related, '
                '.sidebar, .newsletter, footer, .breadcrumb'
            ):
                unwanted.decompose()

            article["content_html"] = str(content_elem)
            article["content_markdown"] = self.html_to_markdown(str(content_elem))

        # 自动分类
        title_lower = article["title"].lower()
        url_lower = url.lower()

        disease_keywords = [
            'pancreatitis', 'stone', 'allerg', 'infection', 'cataract',
            'luxat', 'hypothyroid', 'diabetes', 'cushing', 'dental disease',
            'seizure', 'liver disease', 'heart', 'cancer', 'tumor',
        ]
        prevention_keywords = ['vaccin', 'deworm', 'spay', 'neuter']
        nutrition_keywords = ['nutrition', 'feeding', 'diet', 'food', 'weight']
        grooming_keywords = ['groom', 'coat', 'dental care', 'ear clean', 'nail']

        if any(kw in title_lower or kw in url_lower for kw in disease_keywords):
            article["metadata"]["category"] = "健康医疗/常见疾病"
        elif any(kw in title_lower or kw in url_lower for kw in prevention_keywords):
            article["metadata"]["category"] = "健康医疗/预防保健"
        elif any(kw in title_lower or kw in url_lower for kw in nutrition_keywords):
            article["metadata"]["category"] = "饮食营养"
        elif any(kw in title_lower or kw in url_lower for kw in grooming_keywords):
            article["metadata"]["category"] = "美容护理"
        else:
            article["metadata"]["category"] = "综合"

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

        if article["title"] and article.get("content_markdown") and len(article["content_markdown"]) > 200:
            self.save_article(url, article)
        else:
            self.logger.warning(f"无法提取有效内容: {url}")

    def _crawl_search(self):
        for keyword in self.search_keywords:
            self.logger.info(f"搜索: {keyword}")
            search_url = f"{self.base_url}/search?search_api_fulltext={quote_plus(keyword)}"

            html = self.fetch_page(search_url)
            if not html:
                continue

            soup = self.parse_html(html)
            links = self.get_links_from_page(
                soup, self.base_url,
                filter_fn=self._is_relevant_url
            )

            self.logger.info(f"  搜索 '{keyword}' 找到 {len(links)} 个链接")
            for link in links[:10]:
                self._crawl_article(link)

    def crawl(self):
        self.logger.info("--- 阶段 1: 爬取直接目标页面 ---")
        for url in self.direct_urls:
            self._crawl_article(url)

        self.logger.info("--- 阶段 2: 搜索补充内容 ---")
        self._crawl_search()


if __name__ == "__main__":
    crawler = VCACrawler()
    crawler.run()