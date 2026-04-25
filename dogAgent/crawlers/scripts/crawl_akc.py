"""
AKC (American Kennel Club) 爬虫
爬取雪纳瑞品种页面及相关专家建议文章
"""

import re
from urllib.parse import urljoin, urlparse
from base_crawler import BaseCrawler


class AKCCrawler(BaseCrawler):
    """AKC 网站爬虫"""

    def __init__(self):
        super().__init__("akc")
        self.base_url = "https://www.akc.org"
        # 雪纳瑞相关的直接文章 URL
        self.schnauzer_urls = [
            # 品种页面
            "https://www.akc.org/dog-breeds/miniature-schnauzer/",
            "https://www.akc.org/dog-breeds/standard-schnauzer/",
            "https://www.akc.org/dog-breeds/giant-schnauzer/",
        ]
        # 通用养犬知识文章的搜索关键词
        self.search_keywords = self.source_config.get("search_keywords", [])

    def _is_relevant_url(self, url: str) -> bool:
        """判断 URL 是否是我们需要的文章页面"""
        parsed = urlparse(url)
        path = parsed.path.lower()

        # 排除非文章页面
        exclude_patterns = [
            '/shop/', '/marketplace/', '/register/', '/login/',
            '/events/', '/sports/', '/clubs/', '/products/',
            '/wp-content/', '/wp-admin/', '/cart/', '/account/',
            '.pdf', '.jpg', '.png', '.gif', '.mp4',
        ]
        for pattern in exclude_patterns:
            if pattern in path:
                return False

        # 包含的页面类型
        include_patterns = [
            '/expert-advice/',
            '/dog-breeds/miniature-schnauzer',
            '/dog-breeds/standard-schnauzer',
            '/dog-breeds/giant-schnauzer',
        ]
        for pattern in include_patterns:
            if pattern in path:
                return True

        return False

    def _is_schnauzer_related(self, text: str) -> bool:
        """判断文章内容是否与雪纳瑞相关或通用养犬知识相关"""
        text_lower = text.lower()
        # 直接提到雪纳瑞
        schnauzer_keywords = ['schnauzer', 'schnauzers']
        for kw in schnauzer_keywords:
            if kw in text_lower:
                return True
        # 通用养犬知识（健康、营养、训练等）也保留
        general_keywords = [
            'pancreatitis', 'bladder stone', 'urinary', 'skin allerg',
            'ear infection', 'dental', 'vaccination', 'deworm',
            'nutrition', 'diet', 'feeding', 'toxic food', 'poison',
            'grooming', 'puppy training', 'house training',
            'separation anxiety', 'barking', 'socialization',
            'first aid', 'emergency', 'senior dog',
        ]
        for kw in general_keywords:
            if kw in text_lower:
                return True
        return False

    def extract_article_content(self, soup, url=""):
        """提取 AKC 文章内容"""
        article = {
            "title": "",
            "content_html": "",
            "content_markdown": "",
            "author": "",
            "date": "",
            "tags": [],
            "metadata": {}
        }

        # 提取标题
        title_tag = soup.find('h1')
        if title_tag:
            article["title"] = title_tag.get_text(strip=True)

        # 提取作者
        author_tag = soup.find('span', class_=re.compile(r'author|byline', re.I))
        if not author_tag:
            author_tag = soup.find('a', class_=re.compile(r'author', re.I))
        if author_tag:
            article["author"] = author_tag.get_text(strip=True)

        # 提取日期
        date_tag = soup.find('time')
        if not date_tag:
            date_tag = soup.find(class_=re.compile(r'date|publish', re.I))
        if date_tag:
            article["date"] = date_tag.get_text(strip=True)

        # 提取正文内容
        # AKC 文章内容通常在 article 或特定 class 的 div 中
        content_selectors = [
            'article .entry-content',
            'article .article-content',
            '.article-body',
            '.entry-content',
            'article',
            '.page-content',
            'main .content',
        ]

        content_elem = None
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                break

        if content_elem:
            # 移除不需要的元素
            for unwanted in content_elem.select(
                'script, style, nav, footer, .ad, .advertisement, '
                '.social-share, .related-articles, .sidebar, '
                '.newsletter-signup, .comments, .breadcrumb'
            ):
                unwanted.decompose()

            article["content_html"] = str(content_elem)
            article["content_markdown"] = self.html_to_markdown(str(content_elem))

        # 提取标签
        tag_elements = soup.select('.tag a, .tags a, .article-tags a, .category a')
        article["tags"] = [tag.get_text(strip=True) for tag in tag_elements]

        # 判断页面类型
        if '/dog-breeds/' in url:
            article["metadata"]["category"] = "品种百科"
            # 品种页面可能有特殊结构（特性表格等）
            breed_info = self._extract_breed_info(soup)
            if breed_info:
                article["metadata"]["breed_info"] = breed_info
        elif '/expert-advice/health/' in url:
            article["metadata"]["category"] = "健康医疗"
        elif '/expert-advice/nutrition/' in url:
            article["metadata"]["category"] = "饮食营养"
        elif '/expert-advice/training/' in url:
            article["metadata"]["category"] = "训练与行为"
        elif '/expert-advice/dog-breeding/' in url:
            article["metadata"]["category"] = "繁殖与幼犬"
        elif '/expert-advice/lifestyle/' in url:
            article["metadata"]["category"] = "日常饲养"
        else:
            article["metadata"]["category"] = "综合"

        return article

    def _extract_breed_info(self, soup):
        """提取品种信息表格"""
        breed_info = {}

        # AKC 品种页面通常有特性评分
        trait_items = soup.select('.breed-trait-group__trait, .breed-trait-score')
        for item in trait_items:
            label = item.select_one('.breed-trait-group__trait-label, .trait-label')
            score = item.select_one('.breed-trait-score__score, .trait-score')
            if label and score:
                breed_info[label.get_text(strip=True)] = score.get_text(strip=True)

        # 基本信息（身高、体重等）
        vital_stats = soup.select('.breed-vital-stats__stat, .breed-page__hero-stat')
        for stat in vital_stats:
            label = stat.select_one('.breed-vital-stats__stat-label, dt')
            value = stat.select_one('.breed-vital-stats__stat-value, dd')
            if label and value:
                breed_info[label.get_text(strip=True)] = value.get_text(strip=True)

        return breed_info if breed_info else None

    def _crawl_search_results(self):
        """通过站内搜索爬取相关文章"""
        for keyword in self.search_keywords:
            self.logger.info(f"搜索关键词: {keyword}")
            search_url = f"{self.base_url}/search/?q={keyword.replace(' ', '+')}"

            html = self.fetch_page(search_url)
            if not html:
                continue

            soup = self.parse_html(html)

            # 提取搜索结果中的文章链接
            article_links = self.get_links_from_page(
                soup, self.base_url,
                selector='a',
                filter_fn=self._is_relevant_url
            )

            self.logger.info(f"  找到 {len(article_links)} 个相关链接")

            for link in article_links[:20]:  # 每个关键词最多爬20个
                if self.is_crawled(link):
                    self.logger.info(f"  跳过已爬取: {link}")
                    self.stats["skipped"] += 1
                    continue

                self._crawl_article(link)

    def _crawl_article(self, url: str):
        """爬取单篇文章"""
        html = self.fetch_page(url)
        if not html:
            return

        # 保存原始 HTML
        self.save_raw_html(url, html)

        soup = self.parse_html(html)
        article = self.extract_article_content(soup, url)

        if article["title"] and article["content_markdown"]:
            # 检查内容长度，太短的可能不是正文
            if len(article["content_markdown"]) > 200:
                self.save_article(url, article)
            else:
                self.logger.warning(f"  内容过短，跳过: {url}")
        else:
            self.logger.warning(f"  无法提取内容: {url}")

    def _crawl_expert_advice_index(self):
        """爬取专家建议分类页面的文章列表"""
        categories = [
            "health", "nutrition", "training", "lifestyle", "dog-breeding"
        ]

        for category in categories:
            self.logger.info(f"爬取分类: {category}")
            page = 1
            max_pages = 5  # 每个分类最多爬5页

            while page <= max_pages:
                if page == 1:
                    index_url = f"{self.base_url}/expert-advice/{category}/"
                else:
                    index_url = f"{self.base_url}/expert-advice/{category}/page/{page}/"

                html = self.fetch_page(index_url)
                if not html:
                    break

                soup = self.parse_html(html)

                # 提取文章链接
                article_links = self.get_links_from_page(
                    soup, self.base_url,
                    selector='a',
                    filter_fn=lambda u: '/expert-advice/' in u and u != index_url
                )

                if not article_links:
                    break

                self.logger.info(f"  第 {page} 页找到 {len(article_links)} 个链接")

                for link in article_links:
                    if self.is_crawled(link):
                        self.stats["skipped"] += 1
                        continue

                    # 先获取页面检查是否与我们的主题相关
                    article_html = self.fetch_page(link)
                    if not article_html:
                        continue

                    # 检查相关性
                    if self._is_schnauzer_related(article_html):
                        self.save_raw_html(link, article_html)
                        article_soup = self.parse_html(article_html)
                        article = self.extract_article_content(article_soup, link)
                        if article["title"] and len(article.get("content_markdown", "")) > 200:
                            self.save_article(link, article)
                    else:
                        self.logger.debug(f"  不相关，跳过: {link}")

                page += 1

    def crawl(self):
        """主爬取流程"""
        # 1. 爬取雪纳瑞品种页面
        self.logger.info("--- 阶段 1: 爬取雪纳瑞品种页面 ---")
        for url in self.schnauzer_urls:
            if self.is_crawled(url):
                self.logger.info(f"跳过已爬取: {url}")
                self.stats["skipped"] += 1
                continue
            self._crawl_article(url)

        # 2. 爬取专家建议分类页面
        self.logger.info("--- 阶段 2: 爬取专家建议文章 ---")
        self._crawl_expert_advice_index()

        # 3. 通过搜索爬取更多相关文章
        self.logger.info("--- 阶段 3: 搜索相关文章 ---")
        self._crawl_search_results()


if __name__ == "__main__":
    crawler = AKCCrawler()
    crawler.run()