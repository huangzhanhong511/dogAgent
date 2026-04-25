.PHONY: test lint lint-report lint-prune lint-crossref lint-all rebuild-index accuracy-test serve

# ── 测试 ──
test:
	cd dogAgent && .venv/bin/python -m pytest tests/ -q

accuracy-test:
	cd dogAgent && .venv/bin/python tests/test_knowledge_accuracy.py --quick

accuracy-test-full:
	cd dogAgent && .venv/bin/python tests/test_knowledge_accuracy.py

# ── 知识库维护 ──
lint-report:
	cd dogAgent && .venv/bin/python agent/wiki_lint.py --report

lint-prune:
	cd dogAgent && .venv/bin/python agent/wiki_lint.py --prune
	cd dogAgent && .venv/bin/python agent/build_index.py

lint-crossref:
	cd dogAgent && .venv/bin/python agent/wiki_lint.py --crossref

lint-all: lint-prune lint-crossref rebuild-index
	@echo "✅ 知识库维护完成"

lint: lint-report

# ── 索引 ──
rebuild-index:
	cd dogAgent && .venv/bin/python agent/build_index.py

# ── 知识库构建 ──
crawl:
	cd dogAgent/crawlers/scripts && ../../../dogAgent/.venv/bin/python run_all.py

clean-data:
	cd dogAgent/crawlers/scripts && ../../../dogAgent/.venv/bin/python clean_and_organize.py

generate-topics:
	cd dogAgent && .venv/bin/python agent/generate_topics.py

build-wiki:
	cd dogAgent && .venv/bin/python agent/build_wiki.py

# ── 完整知识库流水线 ──
knowledge-pipeline: crawl clean-data generate-topics build-wiki rebuild-index lint-all
	@echo "✅ 知识库完整流水线完成"

# ── 服务 ──
serve:
	cd dogAgent && .venv/bin/python -m uvicorn api.server:app --reload --port 8000
