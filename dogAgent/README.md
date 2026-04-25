# dogAgent — 雪纳瑞养护知识助手

基于 **Karpathy LLM Wiki** + **Lossless Context Management (LCM)** 的雪纳瑞犬知识问答系统。

## 核心特性

- **LLM Wiki 知识库** — 85 篇雪纳瑞专业文章，覆盖品种、饮食、健康、美容、训练等 9 个分类
- **两层 LLM 索引检索** — 不用向量数据库，LLM 读精简索引判断相关文章（准确度 9.6/10）
- **DAG 记忆压缩** — 对话历史用 DAG 结构压缩，但任何信息都不会丢失，可通过 tool calling 随时回溯原文
- **多用户多会话** — JWT 认证，每个用户独立的会话和记忆
- **用户偏好记忆** — LLM 自动从对话中提取用户信息，跨会话持久记忆

## 项目架构

```
dogAgent/
├── agent/                      # 核心 Agent 模块
│   ├── chat.py                     # 对话引擎（检索 + 记忆 + LLM）
│   ├── memory.py                   # 记忆系统（5 张表 + LCM context_items）
│   ├── compaction.py               # DAG 压缩引擎（leaf + condensation）
│   ├── memory_drilldown.py         # 无损回溯（Lossless Claw drill-down）
│   ├── session.py                  # 会话管理（多会话 + 自动标题）
│   ├── background.py               # 后台任务（偏好提取/压缩/标题）
│   ├── query_rewrite.py            # 上下文感知查询重写
│   ├── retriever.py                # 检索器（两层 LLM 索引 + 规则 fallback）
│   ├── build_wiki.py               # Wiki 生成（多源合并 + 同义词分组）
│   ├── build_index.py              # 两层索引生成
│   ├── generate_topics.py          # LLM 补缺生成
│   ├── wiki_lint.py                # 知识库维护（审计/清理/交叉引用）
│   └── sync_to_obsidian.py         # Obsidian 同步
├── api/                        # REST API
│   ├── server.py                   # FastAPI 应用
│   ├── auth.py                     # JWT 认证
│   └── models.py                   # 请求/响应模型
├── web/                        # Web UI
│   └── index.html                  # 单文件 SPA
├── wiki/                       # LLM 生成的 Wiki（85 篇）
│   ├── index.md                    # 顶层索引（~1KB）
│   ├── index.json                  # 检索索引
│   ├── 01-品种百科/
│   ├── 02-饮食营养/
│   ├── 03-健康医疗/
│   ├── 04-美容护理/
│   ├── 05-训练与行为/
│   ├── 06-日常饲养/
│   ├── 07-繁殖与幼犬/
│   ├── 08-法规与养犬常识/
│   └── 09-参考资料/
├── knowledge/                  # 原始知识源（爬取 + LLM 生成）
├── crawlers/                   # 爬虫（PetMD/AKC/VCA/ASPCA/Boqii/Reddit）
│   ├── config/
│   │   ├── sources.json            # 爬虫源配置
│   │   └── target_topics.json      # 52 个目标主题清单
│   └── scripts/
├── tests/                      # 测试
│   ├── test_memory.py              # 记忆系统单元测试（24 项）
│   ├── test_compaction.py          # 压缩引擎单元测试（13 项）
│   ├── test_e2e.py                 # 端到端集成测试（25 项）
│   ├── test_api.py                 # API 层测试（21 项）
│   └── test_knowledge_accuracy.py  # 知识准确度测试（15 题，9.6/10）
├── helpMd/                     # 技术文档
│   ├── 01-系统架构.md
│   ├── 02-记忆系统实现.md
│   └── 03-知识库实现.md
├── scripts/
│   └── weekly_lint.sh              # 知识库定期维护脚本
├── Makefile                    # 统一命令入口
└── data/                       # 运行时数据（SQLite）
```

## 快速开始

### 1. 安装依赖

```bash
cd dogAgent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入:
#   LLM_PROVIDER=qwen 或 openai
#   CHAT_MODEL=gpt-4o
#   OPENAI_API_KEY=xxx（如果用 openai）
```

### 3. 启动服务

```bash
# 方式一: Makefile
make serve

# 方式二: 直接启动
cd dogAgent
python -m uvicorn api.server:app --reload --port 8000
```

访问 http://localhost:8000 使用 Web UI，http://localhost:8000/docs 查看 API 文档。

### 4. CLI 对话

```bash
python agent/chat.py              # 交互模式（带记忆）
python agent/chat.py --no-memory  # 无记忆模式
python agent/chat.py "雪纳瑞胰腺炎怎么治？"  # 单次查询
```

CLI 命令：`/new` 新建会话 | `/sessions` 会话列表 | `/switch <id>` 切换 | `/prefs` 查看偏好 | `debug` 调试模式

## 数据流

```
用户输入
  ↓
查询重写（QueryRewriter: 补全代词和上下文）
  ↓
两层 LLM 检索
  ├─ Step 1: LLM 读 wiki/index.md (~1KB) → 选分类
  └─ Step 2: LLM 读 wiki/{分类}/index.md → 选文章 → 加载内容
  ↓
记忆上下文组装
  ├─ context_items 中的 summary → XML 注入 system prompt
  ├─ context_items 中的 message → 对话历史（fresh tail）
  └─ user_preferences → 用户偏好注入 system prompt
  ↓
LLM 调用（支持 tool calling: memory_expand 展开历史摘要）
  ↓
持久化 + 后台任务
  ├─ 保存消息到 conversations + context_items
  ├─ [后台] LLM 提取用户偏好
  ├─ [后台] 检查并触发 DAG 压缩
  └─ [后台] 自动生成会话标题
```

## API 端点

| 方法 | 路径 | 说明 | 认证 |
|------|------|------|------|
| POST | `/api/auth/register` | 注册 | - |
| POST | `/api/auth/login` | 登录 | - |
| GET | `/api/auth/me` | 当前用户 | JWT |
| POST | `/api/chat` | 对话（检索+记忆+LLM） | JWT |
| GET | `/api/sessions` | 会话列表 | JWT |
| POST | `/api/sessions` | 新建会话 | JWT |
| GET | `/api/sessions/{id}/messages` | 会话消息（context_items 视图） | JWT |
| GET | `/api/preferences` | 用户偏好 | JWT |
| POST | `/api/preferences` | 添加偏好 | JWT |
| GET | `/api/health` | 健康检查 | - |

## 知识库管理

### 完整构建流水线

```bash
make knowledge-pipeline  # 爬虫 → 清洗 → LLM补缺 → Wiki生成 → 索引 → 维护
```

或分步执行：

```bash
make crawl            # 1. 爬取（PetMD/AKC/VCA/ASPCA 等）
make clean-data       # 2. 清洗整理到 knowledge/
make generate-topics  # 3. LLM 补缺（对比 target_topics.json）
make build-wiki       # 4. Wiki 生成（多源合并 + 冲突检测）
make rebuild-index    # 5. 重建两层索引
make lint-all         # 6. 维护（清理无关文章 + LLM 交叉引用）
```

### 知识库维护

```bash
make lint             # 审计报告
make lint-all         # 完整维护（清理 + 交叉引用 + 重建索引）

# 或直接调用
python agent/wiki_lint.py --report       # 审计报告
python agent/wiki_lint.py --prune        # 清理无关文章（移到 _archive/）
python agent/wiki_lint.py --crossref     # LLM 语义交叉引用
python agent/wiki_lint.py --fix-degraded # 修复降级文章
```

定期维护：
```bash
# crontab: 每周一凌晨 3 点
0 3 * * 1 /path/to/scripts/weekly_lint.sh
```

## 测试

```bash
make test              # 65 个单元/集成测试
make accuracy-test     # 快速准确度测试（5 题）
make accuracy-test-full # 完整准确度测试（15 题）
```

测试覆盖：

| 测试文件 | 数量 | 内容 |
|----------|------|------|
| test_memory.py | 24 | 记忆系统：对话存储、偏好、DAG、context_items |
| test_compaction.py | 13 | 压缩引擎：fresh tail、chunk 选择、端到端压缩 |
| test_e2e.py | 25 | 端到端：对话流程、记忆上下文、会话管理、命令处理 |
| test_api.py | 21 | API 层：认证、对话、会话隔离、偏好隔离、压缩触发 |
| test_knowledge_accuracy.py | 15 | 准确度：品种/饮食/健康/美容/训练 全覆盖，平均 9.6/10 |

## 记忆系统关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| context_budget | 8000 token | 上下文总预算 |
| context_threshold | 0.75 | 压缩触发阈值（6000 token） |
| fresh_tail_count | 8 | 保护最后 N 条消息 |
| leaf_target_tokens | 600 | 叶子摘要目标大小 |
| condensed_target_tokens | 900 | 浓缩摘要目标大小 |
| min_fanout | 6 | 触发 condensation 的最小摘要数 |
| STALE_DAYS | 30 | 偏好过期天数 |

## 技术栈

| 层 | 技术 |
|----|------|
| 前端 | HTML/CSS/JS + marked.js |
| API | FastAPI + uvicorn |
| LLM | LangChain + Qwen（支持 Qwen / OpenAI） |
| 数据库 | SQLite |
| 认证 | JWT (PyJWT + bcrypt) |
| 爬虫 | requests + BeautifulSoup4 + html2text |
| 测试 | pytest + FastAPI TestClient |
| 运行 | Python 3.10+ |

## 技术文档

详细的设计文档和面试准备材料在 `helpMd/` 目录：

- [系统架构](helpMd/01-系统架构.md) — 架构全景、数据流、问题解决方案
- [记忆系统实现](helpMd/02-记忆系统实现.md) — LCM 设计、DAG 压缩算法、无损回溯
- [知识库实现](helpMd/03-知识库实现.md) — LLM Wiki 理念、两层索引、维护流程
