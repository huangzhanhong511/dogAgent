# LLM Wiki 演进路线图

> 版本：v1.0 | 日期：2026-04-23
> 
> 灵感来源：[Karpathy — LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)

---

## 1. Karpathy LLM Wiki 核心思想

### 传统 RAG vs LLM Wiki

| 维度 | 传统 RAG | LLM Wiki |
|------|---------|----------|
| 知识存储 | 原始文档片段 | LLM 预处理的结构化 Wiki |
| 查询时行为 | 每次从头检索 + 合成 | 从已合成的 Wiki 中检索 |
| 知识积累 | ❌ 不积累，每次重新发现 | ✅ 持续积累，越用越丰富 |
| 多源合成 | 查询时临时拼接 5 篇文档 | Wiki 中已经综合好了 |
| 维护成本 | 低（只存不管） | 高（但 LLM 做维护） |

### 核心理念

> **Wiki 是一个持续积累的制品（persistent, compounding artifact）。**
> 交叉引用已经建好，矛盾已经被标记，综合分析已经反映了你读过的所有内容。

### 三层架构

```
┌─────────────────────────────────────────────┐
│  Raw Sources（原始源文件）                     │
│  不可变，LLM 只读不改。来源的真相。             │
├─────────────────────────────────────────────┤
│  The Wiki（LLM 生成的 Wiki）                  │
│  结构化的 Markdown 文件集合。                   │
│  LLM 拥有这一层：创建、更新、维护交叉引用。      │
│  你读它；LLM 写它。                            │
├─────────────────────────────────────────────┤
│  The Schema（规则文档）                        │
│  告诉 LLM 如何维护 Wiki 的约定和工作流。         │
│  你和 LLM 共同演化这个文档。                    │
└─────────────────────────────────────────────┘
```

### 三种核心操作

| 操作 | 描述 |
|------|------|
| **Ingest** | 新增源文件 → LLM 读取 → 讨论要点 → 写摘要页 → 更新索引 → 更新相关实体/概念页 → 一个源可能触达 10-15 个 Wiki 页面 |
| **Query** | 从 Wiki 检索 → LLM 合成回答 → **好答案回写为新 Wiki 页** |
| **Lint** | 定期健康检查：矛盾、过时信息、孤立页、缺失链接、重要概念缺少独立页面 |

### 索引和日志

| 文件 | 作用 |
|------|------|
| **index.md** | 内容导向的目录。每页一行摘要，按分类组织。LLM 查询时先读 index 找相关页。 |
| **log.md** | 时间线操作日志。每次 ingest/query/lint 追加一条记录。 |

### 检索策略

Karpathy 的观点：
- **小规模（~100 源，数百页）**：index 文件 + LLM 读取就够了，不需要 embedding
- **大规模**：推荐 [qmd](https://github.com/tobi/qmd)（本地混合 BM25/向量搜索 + LLM 重排序）

### 为什么这能工作

> 维护知识库的烦人部分不是阅读和思考——是记账。更新交叉引用、保持摘要最新、标注新旧数据矛盾、在几十个页面间保持一致性。人类会放弃 Wiki 因为维护负担增长比价值快。LLM 不会厌倦、不会忘记更新一个交叉引用、能一次修改 15 个文件。

---

## 2. dogAgent 当前实现对比

### 三层架构

| Karpathy 层 | dogAgent 对应 | 状态 | 说明 |
|-------------|--------------|------|------|
| Raw Sources | `knowledge/` | ✅ | 爬虫采集 + 清洗后的 Markdown，不可变 |
| The Wiki | `wiki/` | ✅ | LLM 生成的结构化 Wiki（中英双语） |
| The Schema | `build_wiki.py` 中的 System Prompt | ⚠️ | 硬编码在代码里，不是独立的 schema 文件 |

### 核心操作

| 操作 | Karpathy 设计 | dogAgent 现状 | 差距 |
|------|-------------|--------------|------|
| **Ingest** | 增量：新源 → 更新多个 Wiki 页 | 批量：扫描 knowledge/ → 一次性生成所有 Wiki | 🔴 不能增量更新 |
| **Query** | 检索 Wiki → 回答 → 好答案回写 | 检索 Wiki → 回答 | 🟡 不能回写 |
| **Lint** | 定期健康检查 | 只在生成时检测冲突 | 🟡 不能定期 lint |

### 索引和日志

| 文件 | Karpathy | dogAgent | 差距 |
|------|----------|----------|------|
| index | `index.md`（Markdown 目录） | `index.json`（JSON 多维度索引） | ✅ 功能更丰富 |
| log | `log.md`（时间线日志） | ❌ 没有 | 🔴 缺失 |

### 检索方式

| Karpathy | dogAgent | 差距 |
|----------|----------|------|
| 小规模：index 文件够用 | `index.json` + 规则匹配 | ✅ 当前够用 |
| 大规模：推荐 qmd 混合搜索 | 无向量检索 | 🟡 未来需要 |

### 数据规模

| 指标 | 当前值 | 备注 |
|------|--------|------|
| 原始数据源 | 3 篇（仅 PetMD） | 6 个爬虫已写好但未运行 |
| Wiki 条目 | 6 个文件（3 主题 × 中英） | 迷你雪纳瑞、标准雪纳瑞、犬白内障 |
| 检索索引 | 6 条 | index.json |

---

## 3. 演进路线图

### Phase 1：增量 Ingest（🔴 最高优先）

**目标**：新增源文件时，LLM 能智能更新已有的 Wiki 页面，而不是重建全部。

**当前问题**：
```
现在: build_wiki.py 扫描全部 → 按主题分组 → 一次性生成
缺陷: 新增 1 篇文章也要重新处理所有文章
```

**目标流程**：
```
新增: knowledge/03-健康医疗/胰腺炎.md
  ↓
incremental_ingest.py:
  1. 检测到新文件（对比 wiki/ 中已有的 source_files）
  2. LLM 读取新文章
  3. 判断需要的操作：
     a. 创建新 Wiki 页: wiki/03-健康医疗/犬胰腺炎.md
     b. 更新已有页: wiki/01-品种百科/迷你雪纳瑞.md（在健康问题章节补充信息）
  4. 更新 index.json
  5. 追加 log.md
```

**涉及的文件**：
- 新增 `agent/incremental_ingest.py`
- 修改 `agent/build_index.py`（支持增量更新 index.json）
- 新增 `wiki/log.md`

### Phase 2：混合检索（🟡 中优先）

**目标**：向量语义检索 + 规则关键词检索的混合，提升语义匹配能力。

**当前问题**：
```
用户: "眼睛有白色的东西" → 规则匹配找不到"白内障"（没有子串包含关系）
```

**目标方案**：
```
wiki/*.md → 分块(chunk) → embedding → FAISS 向量库 (vectorstore/)
                                        ↑ 增量追加，不重建

查询时:
  用户问题 → embedding → FAISS 相似度搜索 → Top-K chunks
                       → 规则匹配 index.json → Top-K articles
                       → RRF 合并 → 最终 Top-K
```

**涉及的文件**：
- 新增 `agent/build_vectors.py`（增量向量化）
- 改造 `agent/retriever.py` → `HybridRetriever`
- 需要 `faiss-cpu` 依赖

**Karpathy 的建议**：小规模时 index 文件就够了。当 Wiki 超过 ~100 页时再加向量搜索。推荐 [qmd](https://github.com/tobi/qmd) 作为本地搜索引擎。

### Phase 3：查询回写 Wiki（🟡 中优先）

**目标**：用户的好问题和 LLM 的好回答可以存回 Wiki，让探索也能积累知识。

**流程**：
```
用户: "迷你雪纳瑞和标准雪纳瑞的健康问题有什么区别？"
LLM: 生成对比分析表格...

用户: /save  (或 LLM 判断这是一个有价值的分析)

→ 创建: wiki/09-分析/迷你vs标准雪纳瑞健康对比.md
→ 更新 index.json
→ 追加 log.md
```

**涉及的文件**：
- `agent/chat.py` 新增 `/save` 命令
- `api/server.py` 新增保存端点

### Phase 4：Wiki Lint（🟢 低优先）

**目标**：定期对 Wiki 做健康检查。

**检查项**：
- 页面间的矛盾信息
- 被新源文件推翻的旧结论
- 孤立页面（没有任何入链）
- 重要概念被提到但没有独立页面
- 缺失的交叉引用
- 可以通过搜索填补的信息空白

**涉及的文件**：
- 新增 `agent/wiki_lint.py`

### Phase 5：Schema 独立化（🟢 低优先）

**目标**：将 Wiki 维护规则从代码中提取到独立的 schema 文档。

```
docs/WIKI_SCHEMA.md:
  - Wiki 目录结构约定
  - 各类条目模板（疾病/品种/营养/训练）
  - 命名规则
  - 交叉引用规则
  - 冲突处理策略
  - 多源合并策略
```

当前这些规则散落在 `build_wiki.py` 的 System Prompt 和 `knowledge/模板/` 中。

### Phase 6：log.md 操作日志（🟢 低优先）

```markdown
# Wiki 操作日志

## [2026-04-18] ingest | PetMD: Miniature Schnauzer
- 创建: wiki/01-品种百科/迷你雪纳瑞.md
- 创建: wiki/01-品种百科/Miniature-Schnauzer.md
- 来源: knowledge/01-品种百科/Miniature-Schnauzer.md

## [2026-04-18] ingest | PetMD: Cataracts in Dogs
- 创建: wiki/03-健康医疗/犬白内障症状原因及治疗.md
- 更新: wiki/01-品种百科/迷你雪纳瑞.md (健康问题章节)
```

---

## 4. 实施进度

### 已完成 ✅

- [x] 三层架构：Raw Sources (knowledge/) + Wiki (wiki/) + Schema (build_wiki.py prompts)
- [x] 爬虫系统：6 个数据源的爬虫脚本
- [x] 批量 Wiki 生成：build_wiki.py（LLM 翻译+结构化+冲突检测）
- [x] 多维度索引：build_index.py → index.json
- [x] 规则检索：retriever.py（关键词+分类+标签+章节+关联）
- [x] Obsidian 同步：sync_to_obsidian.py
- [x] 记忆系统：Lossless Claw DAG + 偏好 + 钻取 + Decay + 异步后台
- [x] 对话系统：chat.py + server.py + Web UI + 认证

### 待实现

- [ ] **Phase 1: 增量 Ingest** — 新源文件智能更新已有 Wiki 页
- [ ] **Phase 2: 混合检索** — 向量语义 + 规则关键词 + RRF 合并
- [ ] **Phase 3: 查询回写** — 好答案存回 Wiki
- [ ] **Phase 4: Wiki Lint** — 定期健康检查
- [ ] **Phase 5: Schema 独立化** — WIKI_SCHEMA.md
- [ ] **Phase 6: log.md** — 操作日志
- [ ] **数据丰富** — 运行更多爬虫，从 6 个数据源采集更多文章

---

## 5. 检索方案深度分析

### 5.1 当前检索方式

`retriever.py` 使用 `index.json` 做多维度规则匹配：

```
用户查询 → 5 个维度加权评分：
  ① 关键词匹配 (0.4): n-gram 子串匹配 keywords 列表
  ② 分类匹配   (0.2): 意图词 → 分类目录映射
  ③ 标签匹配   (0.2): 意图词 → 标签映射
  ④ 关联匹配   (0.1): [[双向链接]]
  ⑤ 章节匹配   (0.1): H2 标题匹配
  + 标题直接匹配 (+0.5)
→ Top-K 排序 → 加载文章内容
```

**核心局限**：纯字符串匹配，不能做语义理解。"眼睛有白色的东西" 匹配不到 "白内障"。

### 5.2 Karpathy 的 index.md 检索设计

Karpathy 的 index 是一个 Markdown 文件，LLM Agent 直接阅读来判断相关性：

```markdown
# Wiki Index

## Entities (实体)
- [[Miniature Schnauzer]] — Small breed from Germany, 12-14 inches. (3 sources)
- [[Standard Schnauzer]] — Medium working dog, medieval ratcatcher. (2 sources)

## Health Conditions (健康状况)
- [[Cataracts in Dogs]] — Eye lens clouding, common in schnauzers. Surgery available. (2 sources)
- [[Pancreatitis]] — Pancreas inflammation, schnauzers predisposed. (1 source)

## Sources (来源摘要)
- [[Source: PetMD Miniature Schnauzer]] — Breed guide. Ingested 2026-04-18.
```

**检索流程**：LLM Agent 读取整个 index.md → 根据用户问题自己判断哪些页面相关 → 用工具打开具体 Wiki 文件 → 阅读后回答。

**关键洞察**：Karpathy 原文说 "This works surprisingly well at moderate scale (~100 sources, ~hundreds of pages) and avoids the need for embedding-based RAG infrastructure"。这是因为 LLM 本身就是最好的"语义检索引擎"——它读每一行摘要时就在做语义理解。

**但注意**：这个方案假设 LLM Agent 可以直接读文件系统（Claude Code / Codex），而我们的 dogAgent LLM 不能读文件，需要程序化检索。

### 5.3 三种候选方案对比

#### 方案 A：LLM 读 Index 判断（Karpathy 方式）

```
用户提问 → 把整个 index 注入 prompt → LLM 阅读判断 → 加载相关页面 → 回答
```

| 优点 | 缺点 |
|------|------|
| ⭐⭐⭐⭐⭐ 检索质量最好（LLM 语义理解） | 每次查询消耗大量 token（index 全文注入） |
| "眼睛白东西" → LLM 看到 "clouding of eye lens" 就懂 | 延迟高（多一轮 LLM 推理） |
| 复杂推理也能做（"适合老年人养吗"） | 成本高（100 页 index ≈ 5K tokens/次） |
| 零基础设施，不需要向量数据库 | 并发差（每个请求多一次 LLM 调用） |
| 实现极简 | 规模上限（500+ 页 index 效果下降） |

**适合场景**：个人使用、单用户、小规模（< 200 页）

#### 方案 B：程序化检索（向量 + 规则）

```
用户提问 → embedding → FAISS 搜索 + 规则匹配 → RRF 合并 → Top-K → 回答
```

| 优点 | 缺点 |
|------|------|
| ⭐⭐⭐⭐⭐ 速度（毫秒级，FAISS 本地） | 语义能力不如 LLM 直接判断 |
| ⭐⭐⭐⭐⭐ 成本极低（embedding ~$0.000002/次） | 需要 FAISS + embedding API |
| ⭐⭐⭐⭐⭐ 并发（本地计算，不阻塞 LLM） | 新文档需要增量向量化 |
| ⭐⭐⭐⭐ 可扩展（1000+ 页无压力） | 复杂推理能力有限 |

**适合场景**：多用户 Web 服务、中大规模（50-1000+ 页）

#### 方案 C：混合两步检索（程序化粗筛 + LLM 精选）

```
用户提问 → 程序化检索 (快速) → Top-10 候选
         → 候选摘要 + 用户问题给 LLM → LLM 精选 Top-3 → 加载页面 → 回答
```

**综合对比：**

| 维度 | A (纯 LLM) | B (纯程序) | C (混合) |
|------|-----------|-----------|---------|
| 检索质量 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 速度 | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 成本/次 | ⭐⭐ (~5K tokens) | ⭐⭐⭐⭐⭐ (~0) | ⭐⭐⭐ (~1K tokens) |
| 并发 | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 可扩展性 | ⭐⭐ (500 页上限) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 实现复杂度 | ⭐⭐⭐⭐⭐ 最简单 | ⭐⭐⭐ | ⭐⭐ |

### 5.4 向量检索技术细节

#### 文档分块（Chunking）

不是对整个文档做一个 embedding，而是分块后对每个小块做 embedding：

```
wiki/01-品种百科/迷你雪纳瑞.md (128 行, ~2600 字)
    ↓ RecursiveCharacterTextSplitter (chunk_size=500, overlap=80)
    ├─ chunk 1: "迷你雪纳瑞起源于德国..."  → embedding [0.021, -0.034, ...]
    ├─ chunk 2: "白内障是晶状体混浊..."    → embedding [0.045, 0.012, ...]
    ├─ chunk 3: "需要均衡的饮食..."        → embedding [-0.018, 0.056, ...]
    └─ ...
```

每个 chunk ~300-600 字符，有重叠防止信息切断。每个 chunk 带 metadata（来源文件、章节、chunk 序号）。

#### 增量向量化

```
新增 wiki/03-健康医疗/胰腺炎.md
  ↓
build_vectors.py:
  1. 检测到新文件（对比已索引文件列表 + hash）
  2. 只对新文件分块 + embedding
  3. FAISS.add_documents() 追加（不重建全部）
  4. FAISS.save_local()

修改已有文件时:
  1. 检测到文件 hash 变化
  2. 删除该文件的所有旧 chunks
  3. 重新分块 + embedding + 追加
```

#### RRF 合并算法

```
RRF_score(doc) = Σ 1/(k + rank_i)    k=60

向量检索: 犬白内障 rank=1, 迷你雪纳瑞 rank=2
规则检索: 犬白内障 rank=1, 迷你雪纳瑞 rank=2

犬白内障 RRF = 1/(60+1) + 1/(60+1) = 0.0328  ← 两边都 Top-1
迷你雪纳瑞 RRF = 1/(60+2) + 1/(60+2) = 0.0323
```

### 5.5 建议的检索演进路径

| 文档规模 | 推荐方案 | 说明 |
|---------|---------|------|
| < 50 篇 | 当前规则匹配 | 不需要改 |
| 50-200 篇 | 方案 B（向量+规则） | 加 FAISS + embedding |
| 追求最优质量 | 方案 C（程序化粗筛+LLM精选） | 多一次 LLM 调用但效果最好 |
| 个人单用户 | 方案 A（Karpathy 方式） | 最简单但不适合多用户并发 |
