"""
PetMD 爬虫
爬取疾病百科、药物信息、饮食营养、护理指南等
PetMD 是最全面的宠物健康信息来源之一
"""

import re
from urllib.parse import urljoin, urlparse, quote_plus
from base_crawler import BaseCrawler


class PetMDCrawler(BaseCrawler):
    """PetMD 网站爬虫"""

    def __init__(self):
        super().__init__("petmd")
        self.base_url = "https://www.petmd.com"
        self.search_keywords = self.source_config.get("search_keywords", [])

        # 直接目标 URL - 雪纳瑞品种页面和重点疾病页面
        self.direct_urls = [
            # 品种页面
            "https://www.petmd.com/dog/breeds/miniature-schnauzer",
            "https://www.petmd.com/dog/breeds/standard-schnauzer",
            # 雪纳瑞高发疾病（PetMD 新 URL 格式）
            "https://www.petmd.com/dog/conditions/eye/c_dg_cataract",
            "https://www.petmd.com/dog/conditions/endocrine/c_multi_hyperlipidemia",
            "https://www.petmd.com/dog/conditions/digestive/c_dg_pancreatitis",
            "https://www.petmd.com/dog/conditions/urinary/c_dg_urolithiasis",
            "https://www.petmd.com/dog/conditions/skin/c_dg_dermatitis_allergic_inhalant",
            "https://www.petmd.com/dog/conditions/ear/c_dg_otitis_externa",
            "https://www.petmd.com/dog/conditions/endocrine/c_dg_hypothyroidism",
            "https://www.petmd.com/dog/conditions/endocrine/diabetes-dogs",
            "https://www.petmd.com/dog/conditions/endocrine/c_dg_hyperadrenocorticism",
            "https://www.petmd.com/dog/conditions/digestive/obesity-dogs",
            "https://www.petmd.com/dog/conditions/musculoskeletal/c_dg_patellar_luxation",
        ]

        # 分类页面 URL
        self.category_urls = {
            "conditions": "https://www.petmd.com/dog/conditions",
            "nutrition": "https://www.petmd.com/dog/nutrition",
            "care": "https://www.petmd.com/dog/care",
            "emergency": "https://www.petmd.com/dog/emergency",
            "puppies": "https://www.petmd.com/dog/puppies",
            "senior": "https://www.petmd.com/dog/senior",
            "behavior": "https://www.petmd.com/dog/behavior",
            "training": "https://www.petmd.com/dog/training",
        }

    def _is_relevant_url(self, url: str) -> bool:
        """判断 URL 是否是相关的文章页面"""
        parsed = urlparse(url)
        path = parsed.path.lower()

        if parsed.netloc != "www.petmd.com":
            return False

        # 排除
        exclude_patterns = [
            '/slideshows/', '/video/', '/author/', '/about/',
            '/contact/', '/terms/', '/privacy/', '/sitemap',
            '.jpg', '.png', '.gif', '.pdf',
        ]
        for p in exclude_patterns:
            if p in path:
                return False

        # 必须是 dog 相关
        if '/dog/' not in path:
            return False

        return True

    def extract_article_content(self, soup, url=""):
        """提取 PetMD 文章内容"""
        article = {
            "title": "",
            "content_html": "",
            "content_markdown": "",
            "author": "",
            "date": "",
            "tags": [],
            "metadata": {}
        }

        # 标题
        title_tag = soup.find('h1')
        if title_tag:
            article["title"] = title_tag.get_text(strip=True)

        # 作者 - PetMD 通常标注兽医审核
        author_section = soup.find(class_=re.compile(r'author|byline|reviewed', re.I))
        if author_section:
            article["author"] = author_section.get_text(strip=True)

        # 日期
        date_tag = soup.find('time')
        if not date_tag:
            date_meta = soup.find('meta', property='article:published_time')
            if date_meta:
                article["date"] = date_meta.get('content', '')
        else:
            article["date"] = date_tag.get('datetime', date_tag.get_text(strip=True))

        # 正文提取 - PetMD 使用 React/Next.js，内容在 <main> 中
        content_selectors = [
            'main',
            'article .field--name-body',
            'article .node__content',
            '.article-content',
            '.field--name-body',
            'article',
        ]

        content_elem = None
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem and len(content_elem.get_text(strip=True)) > 100:
                break

        if content_elem:
            # 清理不需要的元素
            for unwanted in content_elem.select(
                'script, style, nav, header, footer, '
                '.ad, .advertisement, '
                '.social-share, .related-content, .sidebar, '
                '.newsletter, .comments, .breadcrumb'
            ):
                unwanted.decompose()

            article["content_html"] = str(content_elem)
            article["content_markdown"] = self.html_to_markdown(str(content_elem))

        # 提取标签/分类
        tag_elements = soup.select('.tags a, .field--name-field-tags a, .article-tags a')
        article["tags"] = list(set(tag.get_text(strip=True) for tag in tag_elements))

        # 提取"Key Takeaways"或摘要
        summary_elem = soup.select_one('.field--name-field-summary, .article-summary, .key-takeaways')
        if summary_elem:
            article["metadata"]["summary"] = summary_elem.get_text(strip=True)

        # 分类判断
        if '/conditions/' in url:
            article["metadata"]["category"] = "健康医疗/常见疾病"
            # 提取疾病子分类
            if '/digestive/' in url:
                article["tags"].append("消化系统")
            elif '/urinary/' in url:
                article["tags"].append("泌尿系统")
            elif '/skin/' in url:
                article["tags"].append("皮肤")
            elif '/ears/' in url:
                article["tags"].append("耳部")
            elif '/eyes/' in url:
                article["tags"].append("眼部")
            elif '/endocrine/' in url:
                article["tags"].append("内分泌")
            elif '/musculoskeletal/' in url:
                article["tags"].append("骨骼肌肉")
            elif '/neurological/' in url:
                article["tags"].append("神经系统")
            elif '/cardiovascular/' in url:
                article["tags"].append("心血管")
            elif '/mouth/' in url:
                article["tags"].append("口腔")
        elif '/nutrition/' in url:
            article["metadata"]["category"] = "饮食营养"
        elif '/care/' in url:
            article["metadata"]["category"] = "日常饲养"
        elif '/emergency/' in url:
            article["metadata"]["category"] = "健康医疗/急救指南"
        elif '/puppies/' in url:
            article["metadata"]["category"] = "繁殖与幼犬"
        elif '/senior/' in url:
            article["metadata"]["category"] = "健康医疗/老年犬护理"
        elif '/behavior/' in url or '/training/' in url:
            article["metadata"]["category"] = "训练与行为"
        elif '/breeds/' in url:
            article["metadata"]["category"] = "品种百科"
        else:
            article["metadata"]["category"] = "综合"

        return article

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
        article = self.extract_article_content(soup, url)

        if article["title"] and article.get("content_markdown") and len(article["content_markdown"]) > 200:
            self.save_article(url, article)
        else:
            self.logger.warning(f"无法提取有效内容: {url}")

    def _crawl_category(self, category_name: str, category_url: str):
        """爬取分类页面下的文章"""
        self.logger.info(f"爬取分类: {category_name}")

        html = self.fetch_page(category_url)
        if not html:
            return

        soup = self.parse_html(html)

        # 提取文章链接
        links = self.get_links_from_page(
            soup, self.base_url,
            filter_fn=self._is_relevant_url
        )

        self.logger.info(f"  找到 {len(links)} 个链接")

        for link in links:
            self._crawl_article(link)

    def _crawl_search(self):
        """通过搜索获取更多相关文章"""
        for keyword in self.search_keywords:
            self.logger.info(f"搜索: {keyword}")
            search_url = f"{self.base_url}/search?keys={quote_plus(keyword)}"

            html = self.fetch_page(search_url)
            if not html:
                continue

            soup = self.parse_html(html)
            links = self.get_links_from_page(
                soup, self.base_url,
                filter_fn=self._is_relevant_url
            )

            self.logger.info(f"  搜索 '{keyword}' 找到 {len(links)} 个链接")

            for link in links[:15]:  # 每个关键词最多15个
                self._crawl_article(link)

    def crawl(self):
        """主爬取流程"""
        # 1. 爬取直接目标 URL（品种页面 + 重点疾病）
        self.logger.info("--- 阶段 1: 爬取直接目标页面 ---")
        for url in self.direct_urls:
            self._crawl_article(url)

        # 2. 爬取各分类页面
        self.logger.info("--- 阶段 2: 爬取分类页面 ---")
        for name, url in self.category_urls.items():
            self._crawl_category(name, url)

        # 3. 搜索更多相关内容
        self.logger.info("--- 阶段 3: 搜索补充内容 ---")
        self._crawl_search()


if __name__ == "__main__":
    crawler = PetMDCrawler()
    crawler.run()