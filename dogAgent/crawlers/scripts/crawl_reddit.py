"""
Reddit 雪纳瑞社区爬虫

数据源：Reddit r/schnauzers, r/MiniatureSchnauzer, r/dogs
使用 Reddit JSON API（无需认证，但有速率限制）

抓取内容：
- 热门帖子和高赞帖子
- 帖子标题、正文、评论
- 筛选有价值的经验分享帖（健康、饮食、护理、训练相关）
"""

import os
import sys
import json
import time
import logging
import re
from datetime import datetime

import requests
from markdownify import markdownify as md

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from base_crawler import BaseCrawler

logger = logging.getLogger("crawl_reddit")


class RedditCrawler(BaseCrawler):
    """Reddit 雪纳瑞社区爬虫"""

    # Reddit JSON API endpoint（不需要认证）
    REDDIT_BASE = "https://www.reddit.com"

    # 有价值帖子的关键词过滤
    VALUABLE_KEYWORDS = [
        # 健康相关
        "health", "vet", "sick", "disease", "pancreatitis", "stones", "bladder",
        "urinary", "skin", "allergy", "allergies", "ear", "infection", "eyes",
        "cataracts", "diabetes", "cushing", "thyroid", "seizure", "cancer",
        "tumor", "surgery", "dental", "teeth", "limp", "lump", "vomit",
        "diarrhea", "blood", "weight", "obesity", "senior",
        # 饮食相关
        "food", "diet", "feed", "eat", "nutrition", "kibble", "raw", "treat",
        "toxic", "poison", "chocolate", "grape", "onion", "xylitol",
        # 护理相关
        "groom", "grooming", "haircut", "clip", "strip", "stripping", "bath",
        "brush", "nail", "ear clean", "coat", "shed", "matt",
        # 训练相关
        "train", "training", "bark", "barking", "bite", "biting", "aggress",
        "socialize", "puppy", "crate", "potty", "housebreak", "leash",
        "obedience", "command", "behav",
        # 品种特性
        "schnauzer", "mini schnauzer", "miniature schnauzer", "standard schnauzer",
        "giant schnauzer", "breed", "temperament", "personality",
        # 中文关键词（偶尔出现）
        "雪纳瑞",
    ]

    # 需要排除的低价值帖子模式
    EXCLUDE_PATTERNS = [
        r"^\[?\s*photo",
        r"^\[?\s*pic",
        r"^look at",
        r"^just adopted",
        r"^meet my",
        r"^say hello",
        r"^happy birthday",
        r"^rip\b",
        r"^goodnight",
        r"^good morning",
        r"^\[?\s*meme",
    ]

    def __init__(self):
        super().__init__("reddit")
        self.subreddits = self.source_config.get("subreddits", ["schnauzers"])
        self.search_keywords = self.source_config.get("search_keywords", [])
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "dogAgent/1.0 (Schnauzer Knowledge Crawler)",
            "Accept": "application/json",
        })

    def _reddit_get(self, url: str, params: dict = None) -> dict:
        """Reddit API GET 请求，带速率限制"""
        if not url.endswith('.json'):
            url = url.rstrip('/') + '.json'

        time.sleep(2)  # Reddit 要求至少 1 秒间隔

        try:
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()

            # 检查速率限制
            remaining = resp.headers.get('x-ratelimit-remaining')
            reset_time = resp.headers.get('x-ratelimit-reset')
            if remaining and float(remaining) < 5:
                wait = float(reset_time) if reset_time else 60
                logger.warning(f"接近速率限制，等待 {wait:.0f} 秒")
                time.sleep(wait)

            return resp.json()
        except requests.RequestException as e:
            logger.error(f"Reddit API 请求失败: {url} - {e}")
            return {}

    def _is_valuable_post(self, title: str, selftext: str, score: int) -> bool:
        """判断帖子是否有价值"""
        # 低分帖子跳过
        if score < 3:
            return False

        combined = (title + " " + selftext).lower()

        # 排除纯晒图帖
        for pattern in self.EXCLUDE_PATTERNS:
            if re.match(pattern, title.lower().strip()):
                return False

        # 关键词匹配
        for keyword in self.VALUABLE_KEYWORDS:
            if keyword.lower() in combined:
                return True

        # 长文本帖子通常有价值
        if len(selftext) > 500:
            return True

        return False

    def _categorize_post(self, title: str, selftext: str) -> str:
        """根据内容判断帖子分类"""
        combined = (title + " " + selftext).lower()

        health_kw = ["health", "vet", "sick", "disease", "pancreatitis", "stone", "allergy",
                      "infection", "surgery", "cancer", "diabetes", "seizure", "vomit", "blood"]
        food_kw = ["food", "diet", "feed", "eat", "nutrition", "treat", "toxic", "poison", "kibble"]
        groom_kw = ["groom", "haircut", "clip", "strip", "bath", "brush", "nail", "coat"]
        train_kw = ["train", "bark", "bite", "socialize", "puppy", "crate", "potty", "leash", "behav"]

        if any(kw in combined for kw in health_kw):
            return "健康医疗"
        elif any(kw in combined for kw in food_kw):
            return "饮食营养"
        elif any(kw in combined for kw in groom_kw):
            return "美容护理"
        elif any(kw in combined for kw in train_kw):
            return "训练与行为"
        else:
            return "日常饲养"

    def _fetch_top_comments(self, permalink: str, max_comments: int = 10) -> list:
        """获取帖子的高赞评论"""
        url = f"{self.REDDIT_BASE}{permalink}.json"
        data = self._reddit_get(url, params={"sort": "best", "limit": max_comments})

        comments = []
        if not data or len(data) < 2:
            return comments

        comment_listing = data[1].get("data", {}).get("children", [])
        for child in comment_listing[:max_comments]:
            if child.get("kind") != "t1":
                continue
            cdata = child.get("data", {})
            body = cdata.get("body", "")
            score = cdata.get("score", 0)
            author = cdata.get("author", "[deleted]")

            if score < 2 or not body or body == "[deleted]" or body == "[removed]":
                continue

            comments.append({
                "author": author,
                "score": score,
                "body": body,
            })

        return comments

    def _post_to_markdown(self, post: dict, comments: list) -> str:
        """将帖子和评论转为 Markdown"""
        parts = []

        # 正文
        selftext = post.get("selftext", "")
        if selftext and selftext not in ("[deleted]", "[removed]"):
            parts.append(selftext)
        elif post.get("selftext_html"):
            parts.append(md(post["selftext_html"]))

        # 如果是链接帖
        if post.get("is_self") is False and post.get("url"):
            parts.append(f"\n**链接**: {post['url']}\n")

        # 高赞评论
        if comments:
            parts.append("\n## 社区讨论精选\n")
            parts.append("以下为高赞评论，代表社区中有经验的雪纳瑞主人的观点：\n")
            for i, comment in enumerate(comments, 1):
                parts.append(f"### 评论 {i} (👍 {comment['score']})\n")
                parts.append(f"**u/{comment['author']}**:\n")
                parts.append(f"{comment['body']}\n")

        return "\n".join(parts)

    def crawl_subreddit(self, subreddit: str, sort: str = "top", time_filter: str = "all", limit: int = 100):
        """
        爬取一个 subreddit 的帖子

        Args:
            subreddit: 子版块名
            sort: 排序方式 (hot, top, new)
            time_filter: 时间范围 (all, year, month, week)
            limit: 最大帖子数
        """
        logger.info(f"爬取 r/{subreddit} (sort={sort}, time={time_filter})")

        url = f"{self.REDDIT_BASE}/r/{subreddit}/{sort}.json"
        params = {"limit": 100, "t": time_filter}
        after = None
        count = 0

        while count < limit:
            if after:
                params["after"] = after

            data = self._reddit_get(url, params)
            if not data:
                break

            children = data.get("data", {}).get("children", [])
            if not children:
                break

            for child in children:
                if child.get("kind") != "t3":
                    continue

                post = child.get("data", {})
                title = post.get("title", "")
                selftext = post.get("selftext", "")
                score = post.get("score", 0)
                permalink = post.get("permalink", "")
                post_url = f"{self.REDDIT_BASE}{permalink}"

                # 检查是否已爬取
                if self.is_crawled(post_url):
                    continue

                # 过滤低价值帖子
                if not self._is_valuable_post(title, selftext, score):
                    continue

                logger.info(f"  有价值帖子: [{score}👍] {title[:60]}")

                # 获取高赞评论
                comments = self._fetch_top_comments(permalink)

                # 转为 Markdown
                content_md = self._post_to_markdown(post, comments)

                if len(content_md.strip()) < 100:
                    continue

                # 分类
                category = self._categorize_post(title, selftext)

                # 构建文章对象
                article = {
                    "title": title,
                    "content_markdown": content_md,
                    "author": f"u/{post.get('author', '[deleted]')}",
                    "date": datetime.fromtimestamp(post.get("created_utc", 0)).strftime("%Y-%m-%d"),
                    "tags": [f"r/{subreddit}", category, "社区经验"],
                    "metadata": {
                        "reddit_score": score,
                        "num_comments": post.get("num_comments", 0),
                        "subreddit": subreddit,
                        "category": category,
                        "post_type": "经验分享",
                    },
                }

                self.save_article(post_url, article)
                count += 1

                if count >= limit:
                    break

            after = data.get("data", {}).get("after")
            if not after:
                break

        logger.info(f"  r/{subreddit} 完成，共爬取 {count} 篇有价值帖子")
        return count

    def search_subreddit(self, subreddit: str, query: str, limit: int = 50):
        """在 subreddit 中搜索特定关键词"""
        logger.info(f"搜索 r/{subreddit}: '{query}'")

        url = f"{self.REDDIT_BASE}/r/{subreddit}/search.json"
        params = {
            "q": query,
            "restrict_sr": "on",
            "sort": "relevance",
            "t": "all",
            "limit": 100,
        }
        after = None
        count = 0

        while count < limit:
            if after:
                params["after"] = after

            data = self._reddit_get(url, params)
            if not data:
                break

            children = data.get("data", {}).get("children", [])
            if not children:
                break

            for child in children:
                if child.get("kind") != "t3":
                    continue

                post = child.get("data", {})
                title = post.get("title", "")
                selftext = post.get("selftext", "")
                score = post.get("score", 0)
                permalink = post.get("permalink", "")
                post_url = f"{self.REDDIT_BASE}{permalink}"

                if self.is_crawled(post_url):
                    continue

                if not self._is_valuable_post(title, selftext, score):
                    continue

                comments = self._fetch_top_comments(permalink)
                content_md = self._post_to_markdown(post, comments)

                if len(content_md.strip()) < 100:
                    continue

                category = self._categorize_post(title, selftext)

                article = {
                    "title": title,
                    "content_markdown": content_md,
                    "author": f"u/{post.get('author', '[deleted]')}",
                    "date": datetime.fromtimestamp(post.get("created_utc", 0)).strftime("%Y-%m-%d"),
                    "tags": [f"r/{subreddit}", category, "社区经验", query],
                    "metadata": {
                        "reddit_score": score,
                        "num_comments": post.get("num_comments", 0),
                        "subreddit": subreddit,
                        "category": category,
                        "post_type": "经验分享",
                        "search_query": query,
                    },
                }

                self.save_article(post_url, article)
                count += 1

                if count >= limit:
                    break

            after = data.get("data", {}).get("after")
            if not after:
                break

        logger.info(f"  搜索 '{query}' 完成，爬取 {count} 篇")
        return count

    def crawl(self):
        """主爬取入口"""
        total = 0

        # 1. 爬取各 subreddit 的热门和置顶帖
        for subreddit in self.subreddits:
            # Top posts of all time
            total += self.crawl_subreddit(subreddit, sort="top", time_filter="all", limit=100)
            # Top posts of this year
            total += self.crawl_subreddit(subreddit, sort="top", time_filter="year", limit=50)
            # Hot posts (当前热门)
            total += self.crawl_subreddit(subreddit, sort="hot", limit=30)

        # 2. 关键词搜索（在主要 subreddit 中）
        primary_sub = self.subreddits[0] if self.subreddits else "schnauzers"
        for keyword in self.search_keywords:
            total += self.search_subreddit(primary_sub, keyword, limit=20)

        self.logger.info(f"\n===== Reddit 爬取完成，共 {total} 篇有价值帖子 =====")
        return total


def main():
    crawler = RedditCrawler()
    crawler.crawl()


if __name__ == "__main__":
    main()