# dogAgent 记忆系统设计方案

> 版本：v2.1 | 日期：2026-04-19

## 1. 概述

dogAgent 的记忆系统借鉴了 [Lossless Claw (LCM)](https://github.com/Martian-Engineering/lossless-claw) 的 DAG 摘要压缩架构，并结合宠物咨询场景的特点，构建了六大模块。

### 设计目标

- **永不丢失**：所有原始对话消息持久化到 SQLite，任何时候都可以回溯
- **无限对话**：通过 DAG 多层摘要压缩，支持任意长度的对话而不超出 context window
- **用户画像**：自动从对话中提取用户偏好（自然语言存储，无预设 schema），支持级联更新
- **检索增强**：通过 Query Rewrite 提升 RAG 检索精准度
- **记忆检索**：对摘要和偏好生成 embedding，支持语义搜索历史对话
- **会话管理**：支持多个 conversation session，区分不同对话主题
- **多租户隔离**：按 `user_id` 完全隔离对话、偏好和摘要

### 整体架构

```
┌──────────────────────────────────────────────────────────────┐
│                  dogAgent Memory System v2.1                  │
│                                                               │
│  1. Conversation Store  ── 所有原始消息（SQLite）               │
│  2. Summary DAG         ── 多层摘要压缩（不丢失）              │
│  3. User Preferences    ── 自然语言偏好（LLM提取/级联更新）    │
│  4. Query Rewriter      ── 代词消解 + 偏好注入 + 上下文         │
│  5. Memory Embedding    ── 向量化摘要/偏好，语义检索历史记忆    │
│  6. Session Manager     ── 多会话管理（新话题 vs 旧话题）       │
│                                                               │
│  Context Assembly:                                            │
│  [system] + [preferences] + [DAG摘要] + [memory_search]       │
│  + [近期对话] + [RAG(rewritten_query)] + [用户问题]            │
└──────────────────────────────────────────────────────────────┘
```

### 灵感来源

| 系统 | 借鉴点 |
|------|--------|
| [Lossless Claw (LCM)](https://github.com/Martian-Engineering/lossless-claw) | DAG 多层摘要压缩、消息持久化、context assembly 策略 |
| [MemU](https://arxiv.org/abs/2405.xxxxx) | 用户画像提取、自然语言存储、偏好更新机制 |
| RAG 最佳实践 | Query Rewrite 提升检索质量 |

---

## 2. 模块 1：Conversation Store（短期记忆）

### 职责

持久化每一条对话消息，按 `user_id` + `session_id` 隔离，作为 DAG 压缩的数据源。

### 数据库 Schema

```sql
CREATE TABLE conversations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT NOT NULL,
    session_id  TEXT NOT NULL,           -- 所属会话
    role        TEXT NOT NULL,           -- 'user' | 'assistant'
    content     TEXT NOT NULL,
    token_count INTEGER,                 -- 预估 token 数
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_conv_user_session ON conversations(user_id, session_id, created_at);
```

### 关键操作

| 操作 | 说明 |
|------|------|
| `add_message(user_id, session_id, role, content)` | 存储一条消息，自动计算 token_count |
| `get_recent(user_id, session_id, limit=20)` | 获取当前 session 最近 N 条消息（freshTail） |
| `get_range(user_id, start_id, end_id)` | 获取指定范围消息（用于 DAG 回溯） |
| `count_tokens(user_id, session_id)` | 统计当前 session 未压缩消息的 token 总量 |

---

## 3. 模块 2：Summary DAG（长期记忆）

### 核心思想

借鉴 Lossless Claw 的 DAG（有向无环图）摘要结构。当对话 token 超出阈值时，将旧消息分块压缩为摘要，摘要本身积累过多时再浓缩为更高层摘要。

**每个 session 拥有独立的 DAG 树。**

### DAG 结构示意

```
              ┌──────────────────────────────┐
              │    Condensed Summary         │
              │    (depth 2, ~900 tokens)    │
              │    "这是一只3岁雪纳瑞的长期   │
              │     健康咨询，涉及：耳道感染、│
              │     饮食选择、体重管理..."     │
              └──────┬───────────┬───────────┘
                     │           │
         ┌───────────┘           └───────────┐
         ▼                                   ▼
┌─────────────────┐                 ┌─────────────────┐
│  Leaf Summary A  │                 │  Leaf Summary B  │
│  (depth 1)       │                 │  (depth 1)       │
│  消息 #1-#20     │                 │  消息 #21-#40    │
│  ~600 tokens     │                 │  ~600 tokens     │
└────────┬─────────┘                 └────────┬────────┘
         │                                    │
    原始消息 #1-#20                      原始消息 #21-#40
    (保留在 SQLite)                     (保留在 SQLite)
```

### 数据库 Schema

```sql
-- 摘要节点
CREATE TABLE summaries (
    id              TEXT PRIMARY KEY,    -- UUID
    user_id         TEXT NOT NULL,
    session_id      TEXT NOT NULL,       -- 所属会话
    depth           INTEGER NOT NULL,    -- 0=leaf, 1=condensed, 2=super-condensed...
    content         TEXT NOT NULL,       -- 摘要文本
    token_count     INTEGER,
    source_start_id INTEGER,             -- 源消息起始 ID（depth=0 时有效）
    source_end_id   INTEGER,             -- 源消息结束 ID（depth=0 时有效）
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_summary_user_session ON summaries(user_id, session_id, depth);

-- DAG 边：记录摘要的来源关系
CREATE TABLE summary_edges (
    parent_id  TEXT NOT NULL,            -- 高层摘要 ID
    child_id   TEXT NOT NULL,            -- 子摘要 ID 或消息范围标记
    child_type TEXT NOT NULL,            -- 'message_range' | 'summary'
    PRIMARY KEY (parent_id, child_id)
);
```

### 压缩流程

```
触发条件：当前 session 未压缩消息的 token 总量 > context_budget × threshold (0.75)

Step 1: Leaf 压缩 (depth 0)
  - 取最旧的未压缩消息（保留最近 fresh_tail_count 条）
  - 按 leaf_chunk_tokens 分 chunk
  - 每个 chunk 调用 LLM 生成摘要（~600 tokens）
  - 记录 summary + edges
  - 同步生成 embedding → memory_embeddings

Step 2: Condensation (depth N → depth N+1)
  - 当同一 depth 的摘要数量 >= min_fanout (3)
  - 将多个同层摘要浓缩为更高层摘要（~900 tokens）
  - 记录新 summary + edges
  - 同步更新 embedding

Step 3: 重复 Step 2 直到最高层只剩 1-2 个摘要
```

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `context_threshold` | 0.75 | context window 占比阈值，超过即触发压缩 |
| `fresh_tail_count` | 20 | 保护最近 N 条消息不被压缩 |
| `leaf_chunk_tokens` | 4000 | 每个 leaf chunk 的最大 token 数 |
| `leaf_target_tokens` | 600 | leaf 摘要的目标 token 数 |
| `condensed_target_tokens` | 900 | condensed 摘要的目标 token 数 |
| `min_fanout` | 3 | 触发 condensation 的最少子摘要数 |
| `max_rounds` | 5 | 单次压缩的最大循环轮数 |

### Context Assembly（上下文组装）

每轮对话时，按以下顺序组装 prompt：

```
┌─ LLM Prompt ──────────────────────────────────────────┐
│                                                        │
│  1. System Prompt                                      │
│                                                        │
│  2. 用户偏好（跨 session 共享）                         │
│     "关于该用户已知信息：                                │
│      - 用户的狗叫旺旺                                   │
│      - 旺旺是白色迷你雪纳瑞，4岁"                       │
│                                                        │
│  3. DAG 顶层摘要（当前 session 的长期记忆）              │
│     "之前讨论过：饮食选择、体重管理、耳道感染治疗..."     │
│                                                        │
│  4. Memory Search 结果（跨 session 语义检索）            │
│     "在之前的会话中讨论过类似问题：..."                   │
│                                                        │
│  5. 近期 Leaf 摘要（如有）                               │
│     "最近讨论了疫苗接种计划和驱虫时间表"                 │
│                                                        │
│  6. 最近 N 条原始对话（当前 session 的 freshTail）       │
│                                                        │
│  7. RAG 检索结果（用 rewritten query 检索知识库）        │
│                                                        │
│  8. 用户原始问题                                        │
└────────────────────────────────────────────────────────┘
```

### 回溯支持

通过 `summary_edges` 表，可以从任意摘要回溯到原始消息：

```
Condensed Summary → [Leaf A, Leaf B, Leaf C] → 原始消息 #1-#60
```

未来可以实现类似 `lcm_expand` 的功能，让 agent 按需展开摘要查看原文细节。

---

## 4. 模块 3：User Preferences（用户偏好）

### 核心原则

- **自然语言存储**：不预设 schema，每条偏好都是一句自然语言
- **LLM 提取**：每轮对话后由 LLM 判断是否有新偏好可提取
- **级联更新**：当名字等核心属性变化时，所有引用该信息的偏好一起更新
- **跨 Session 共享**：偏好属于用户而非某个话题，所有 session 共享
- **审计追溯**：旧偏好不删除，通过 `superseded_by` 形成更新链

### 数据库 Schema

```sql
CREATE TABLE user_preferences (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       TEXT NOT NULL,
    content       TEXT NOT NULL,          -- 自然语言描述
    source_msg_id INTEGER,                -- 从哪条消息提取的
    source_session_id TEXT,               -- 从哪个 session 提取的
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    superseded_by INTEGER DEFAULT NULL    -- 被更新时指向新条目 ID
);

CREATE INDEX idx_pref_user_active ON user_preferences(user_id, superseded_by);
```

### 提取流程

```
每轮对话后：

1. 将最新一轮对话 + 现有活跃偏好列表发送给 LLM
2. LLM 返回：
   - new:  新偏好条目列表（如果有）
   - update: 需要更新的旧条目 → 新内容（如果有）
3. 执行数据库操作：
   - 插入新条目
   - 标记旧条目 superseded_by = 新条目 ID
   - 为新/更新的偏好生成 embedding → memory_embeddings
```

### 提取 Prompt

```
从以下对话中提取用户关于其宠物或个人偏好的信息。
用自然语言描述，每条一行。

规则：
1. 提取新出现的信息
2. 如果用户更新了已有信息（如改名、年龄变化），标记为 update
3. 级联更新：如果核心属性变化（如狗的名字），所有引用该属性的偏好都要更新
4. 如果没有新信息，返回空

现有偏好：
- 用户的狗叫旺财
- 旺财是白色迷你雪纳瑞
- 旺财3岁

最新对话：
用户：其实我家狗改名叫旺旺了

请返回 JSON：
{
  "new": [],
  "update": [
    {"old": "用户的狗叫旺财", "new": "用户的狗叫旺旺"},
    {"old": "旺财是白色迷你雪纳瑞", "new": "旺旺是白色迷你雪纳瑞"},
    {"old": "旺财3岁", "new": "旺旺3岁"}
  ]
}
如果没有新信息，返回 {"new": [], "update": []}
```

### 级联更新示例

**场景：狗改名 旺财 → 旺旺**

```
之前的偏好：
  id=1  "用户的狗叫旺财"          superseded_by=NULL (活跃)
  id=2  "旺财是白色迷你雪纳瑞"    superseded_by=NULL (活跃)
  id=3  "旺财3岁"                 superseded_by=NULL (活跃)

用户："其实我家狗改名叫旺旺了"

LLM 提取 → 级联更新

更新后：
  id=1  "用户的狗叫旺财"          superseded_by=4    (已过时)
  id=2  "旺财是白色迷你雪纳瑞"    superseded_by=5    (已过时)
  id=3  "旺财3岁"                 superseded_by=6    (已过时)
  id=4  "用户的狗叫旺旺"          superseded_by=NULL (活跃) ← NEW
  id=5  "旺旺是白色迷你雪纳瑞"    superseded_by=NULL (活跃) ← NEW
  id=6  "旺旺3岁"                 superseded_by=NULL (活跃) ← NEW
```

### 审计查询

```sql
-- 查看某条偏好的更新历史
WITH RECURSIVE chain AS (
    SELECT * FROM user_preferences WHERE id = 4
    UNION ALL
    SELECT p.* FROM user_preferences p
    JOIN chain c ON p.superseded_by = c.id
)
SELECT * FROM chain ORDER BY created_at;

-- 结果：
-- id=1 "用户的狗叫旺财"  → id=4 "用户的狗叫旺旺"
```

### 偏好 Decay（过时标注）

长期未被确认的偏好可能已过时（如"3岁"→实际已 4 岁）。通过 `last_confirmed_at` 字段跟踪：

**数据库字段：**
```sql
last_confirmed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
```

**Decay 机制：**
- 每条偏好创建时 `last_confirmed_at = now`
- 每轮对话中偏好被注入 prompt 时，调用 `touch_preferences()` → `last_confirmed_at = now`
- 超过 `STALE_DAYS`（默认 30 天）未确认的偏好，在 prompt 中加标注

**效果：**
```
关于该用户已知信息：
- 用户的狗叫旺旺
- 旺旺是白色的迷你雪纳瑞
- 旺旺3岁  (较久前的信息，可能已过时)
- 用户最近给旺旺换了渴望品牌狗粮  (较久前的信息，可能已过时)
```

LLM 看到 `(较久前的信息，可能已过时)` 标注后，会在回答中主动确认："之前记录旺旺 3 岁，不知道现在是否有变化？"

用户回复确认后 → 偏好被重新注入 prompt → `touch_preferences()` → 恢复为新鲜状态。

### 注入 Prompt

获取活跃偏好（`superseded_by IS NULL`），拼接为文本块注入 system prompt（含 decay 标注）：

```
关于该用户已知信息：
- 用户的狗叫旺旺
- 旺旺是白色的迷你雪纳瑞
- 旺旺3岁
- 用户最近给旺旺换了渴望品牌狗粮
```

---

## 5. 模块 4：Query Rewrite（查询重写）

### 职责

将用户的简短/模糊问题，结合对话上下文和用户偏好，重写为更适合 RAG 检索的查询。

**重写后的 query 仅用于检索（知识库 + 记忆），不替换 prompt 中的用户原始问题。**

### 判断是否需要重写

```python
def needs_rewrite(query, preferences, recent_messages):
    # 1. 包含代词 → 需要
    if any(p in query for p in ['它', '他', '她', '这个', '那个', '上次']):
        return True
    # 2. 问题太短且有上下文 → 需要
    if len(query) < 10 and recent_messages:
        return True
    # 3. 有偏好但问题没提到狗的信息 → 需要
    if preferences and not any(k in query for k in ['雪纳瑞', '狗']):
        return True
    return False
```

### 重写 Prompt

```
你是一个查询重写助手。请将用户的问题改写为更适合知识库检索的查询。

规则：
1. 将代词（它/他/她/这个）替换为具体名词
2. 结合对话上下文补充缺失信息
3. 结合用户偏好补充背景信息
4. 保持简洁，重点突出关键词，适合检索
5. 只输出重写后的查询，不要解释

用户偏好：
{偏好列表}

最近对话：
{最近3轮对话}

用户原始问题：{原始问题}

重写后的检索查询：
```

### 示例

| 原始问题 | 偏好 | 上下文 | 重写结果 |
|----------|------|--------|----------|
| "它最近老掉毛" | 旺旺是迷你雪纳瑞，4岁 | (无) | "4岁迷你雪纳瑞掉毛原因及处理方法" |
| "还需要吃药吗？" | 旺旺是雪纳瑞 | 刚讨论过耳道感染 | "雪纳瑞耳道感染是否需要口服药物治疗" |
| "能洗澡吗？" | 旺旺3个月大，刚接回家 | (无) | "3个月大雪纳瑞幼犬刚到新家是否可以洗澡" |
| "雪纳瑞多大可以绝育？" | (无) | (无) | (不重写，直接检索) |

### 性能

| 项目 | 说明 |
|------|------|
| 额外成本 | 每次重写约 ~200 input tokens + ~30 output tokens |
| 延迟 | ~0.3-0.5s（可配置用 mini 模型） |
| Fallback | 重写失败则用原始问题检索 |

---

## 6. 模块 5：Memory Embedding Index（向量记忆检索）

### 职责

对摘要和偏好生成 embedding 向量，支持**跨 session 语义搜索历史对话**。

当用户在新 session 中提到之前讨论过的话题时，可以从历史记忆中检索相关信息。

### 数据库 Schema

```sql
CREATE TABLE memory_embeddings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT NOT NULL,
    source_type TEXT NOT NULL,           -- 'summary' | 'preference'
    source_id   TEXT NOT NULL,           -- 对应 summaries.id 或 user_preferences.id
    session_id  TEXT,                    -- 来源 session（偏好为 NULL，因为跨 session）
    content     TEXT NOT NULL,           -- 原文本（用于展示）
    embedding   BLOB NOT NULL,           -- numpy float32 array 序列化
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_membed_user ON memory_embeddings(user_id);
```

### 工作方式

**写入时机：**
- 每次生成新的 leaf/condensed 摘要 → 生成 embedding 存入
- 每次新增/更新用户偏好 → 生成 embedding 存入（旧的标记删除）

**检索时机：**
- Query Rewrite 后，用 rewritten query 的 embedding 检索 `memory_embeddings`
- 返回 top-K 最相关的历史记忆片段（排除当前 session 已有的摘要，避免重复）
- 结果作为额外 context 注入 prompt

**检索流程：**
```python
def search_memory(self, user_id: str, query: str, current_session_id: str, top_k: int = 3):
    query_embedding = embed(query)
    
    # 从 memory_embeddings 中检索
    candidates = get_all_embeddings(user_id)
    
    # 排除当前 session 的摘要（已通过 DAG 获取，避免重复）
    candidates = [c for c in candidates if c.session_id != current_session_id or c.source_type == 'preference']
    
    # 计算余弦相似度，返回 top-K
    results = cosine_similarity_topk(query_embedding, candidates, top_k)
    return results
```

### Embedding 模型

使用与 `build_index.py` 相同的模型（OpenAI `text-embedding-3-small`），保持一致性。

### 使用场景

```
Session 1 (上周): 讨论了旺旺的耳道感染治疗
Session 2 (今天): 用户问 "上次那个耳朵的问题好了"

→ Query Rewrite: "旺旺迷你雪纳瑞耳道感染恢复情况"
→ Memory Search: 找到 Session 1 的摘要 "旺旺耳道感染，使用了耳肤灵+洗耳液..."
→ 注入 prompt，LLM 可以引用上次的治疗方案
```

---

## 7. 模块 6：Session Manager（会话管理）

### 职责

管理用户的多个对话会话，区分不同主题。每个 session 拥有独立的对话历史和 DAG 摘要树。

**User Preferences 跨 session 共享**（偏好属于用户，不属于某个话题）。

### 数据库 Schema

```sql
CREATE TABLE sessions (
    id         TEXT PRIMARY KEY,         -- UUID
    user_id    TEXT NOT NULL,
    title      TEXT,                     -- LLM 自动生成的会话标题
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active  BOOLEAN DEFAULT TRUE      -- FALSE = 已归档
);

CREATE INDEX idx_session_user ON sessions(user_id, updated_at DESC);
```

### 会话行为

| CLI 参数 | 行为 |
|----------|------|
| `--session new` | 创建新 session，自动分配 UUID |
| `--session <id>` | 进入指定 session |
| 无 `--session` | 进入最近活跃的 session（`updated_at` 最新） |
| `--session list` | 列出用户所有 session |

### 自动标题

新 session 的前 3 轮对话后，用 LLM 自动生成标题：

```
根据以下对话，生成一个简短的会话标题（10字以内）：
用户：旺旺最近耳朵红红的
助手：可能是耳道感染...

标题：旺旺耳道感染咨询
```

### 数据隔离

```
Session 独有：
  - conversations (按 session_id 过滤)
  - summaries / summary_edges (按 session_id 过滤)
  - memory_embeddings (summary 类型按 session_id)

跨 Session 共享：
  - user_preferences (只按 user_id)
  - memory_embeddings (preference 类型, session_id=NULL)
```

### Session 生命周期

```
创建 → 活跃 (is_active=TRUE) → 归档 (is_active=FALSE)

归档后：
- 不再接收新消息
- 摘要和 embedding 保留（可被 memory search 检索到）
- 偏好不受影响（跨 session）
```

---

## 8. 数据流全景

```
用户输入 "还需要吃药吗？" (session: abc123)
    │
    ├─ 1. SessionManager.ensure_session()       → 确保 session 存在
    │
    ├─ 2. ConversationStore.add_message()       → 持久化消息 (session=abc123)
    │
    ├─ 3. needs_rewrite() → True
    │     QueryRewriter.rewrite()               → "雪纳瑞耳道感染是否需要口服药物"
    │
    ├─ 4. Retriever.retrieve(rewritten_query)   → 检索知识库 (RAG)
    │
    ├─ 5. MemoryIndex.search(rewritten_query)   → 语义检索历史记忆 (跨session)
    │
    ├─ 6. SummaryDAG.get_context(session=abc123) → 获取当前 session 的 DAG 摘要
    │
    ├─ 7. UserPreferences.get_active(user_id)   → 获取用户偏好 (跨session)
    │
    ├─ 8. Context Assembly:
    │     [system_prompt]
    │     + [用户偏好文本]
    │     + [DAG 顶层摘要 (当前session)]
    │     + [Memory Search 结果 (跨session)]
    │     + [近期 leaf 摘要]
    │     + [最近 20 条原始对话 (当前session)]
    │     + [RAG 检索结果]
    │     + [用户原始问题]
    │
    ├─ 9. LLM 生成回答
    │
    ├─ 10. ConversationStore.add_message()      → 持久化回答
    │
    ├─ 11. UserPreferences.extract()            → 提取偏好（级联更新，如有新信息）
    │      + MemoryIndex.upsert_preference()    → 更新偏好 embedding
    │
    └─ 12. Compaction.check_and_compact()       → 检查是否需要 DAG 压缩
           + MemoryIndex.upsert_summary()       → 新摘要生成 embedding
```

---

## 9. 文件结构

```
agent/
├── memory.py            # ConversationStore + UserPreferences + SummaryDAG
├── compaction.py         # DAG 压缩引擎（leaf + condensation + 阈值触发）
├── query_rewrite.py      # QueryRewriter（代词消解 + 偏好注入 + 上下文补全）
├── memory_index.py       # MemoryEmbeddingIndex（向量记忆检索）
├── session.py            # SessionManager（会话管理）
├── chat.py               # 主对话循环（集成全部模块 + 多租户 + session）
├── retriever.py          # RAG 检索（已实现）
├── build_index.py        # 索引构建（已实现）
└── ...
data/
└── memory.db             # SQLite 数据库（所有记忆数据）
```

---

## 10. 多租户隔离

所有数据表都以 `user_id` 为一级隔离键，`session_id` 为二级隔离键：

```
用户 A 的数据：
  sessions WHERE user_id='A'
  conversations WHERE user_id='A' AND session_id='...'
  summaries WHERE user_id='A' AND session_id='...'
  user_preferences WHERE user_id='A'           ← 跨 session
  memory_embeddings WHERE user_id='A'

用户 B 的数据：完全隔离
```

CLI 使用方式：

```bash
# 基本使用（默认用户 + 最近 session）
python agent/chat.py "旺旺最近掉毛了"

# 指定用户
python agent/chat.py --user alice "旺旺最近掉毛了"

# 新建会话
python agent/chat.py --user alice --session new "聊聊绝育的事"

# 进入指定会话
python agent/chat.py --user alice --session abc123 "上次说的耳朵好了吗"

# 列出所有会话
python agent/chat.py --user alice --session list
```

默认 `user_id` 为 `"default"`，默认进入最近活跃的 session。

---

## 11. 模块 7：DAG 回溯钻取（Memory Drill-Down）

### 职责

当用户追问历史对话中的具体细节时（如"上次说的那个药叫什么"），自动从 DAG 摘要树向下钻取到更低层摘要或原始消息，补充高层摘要中缺失的精确信息。

**这是 Lossless Claw "无损"设计的核心体现**：摘要是"有损压缩视图"，原始消息是"无损原始数据"。钻取就是从视图回溯到原始数据。

### 架构位置

```
Condensed Summary (depth 2)   ← get_top_summaries() 注入 prompt
     ↓ drill_down()
Leaf Summary (depth 0)        ← 钻取获得更多细节
     ↓ drill_down()
原始消息 (conversations)      ← 无损原始数据
```

### 触发条件

查询中包含"细节追问"关键词时触发：

| 类别 | 关键词示例 |
|------|-----------|
| 回忆类 | 上次、之前、刚才、你说的、提到过、记得 |
| 细节追问类 | 具体、详细、什么药、什么牌子、叫什么、哪个、多少钱、哪天 |

### 钻取流程

```
用户问："上次说的那个耳药水叫什么？"
  ↓
1. needs_drilldown() → True（"上次" 命中触发词）
  ↓
2. find_relevant_summaries() → 找到最相关的摘要
   "讨论了旺旺的耳道感染治疗方案"
  ↓
3. drill_down() → 通过 DAG 边回溯
   summary_edges: parent → msg_range:1-4
  ↓
4. get_range(user_id, 1, 4) → 加载原始消息
   用户: 旺旺耳朵红红的怎么办
   助手: 建议使用耳肤灵滴耳液，每天两次
   用户: 需要吃消炎药吗
   助手: 可以口服阿莫西林克拉维酸钾，每公斤12.5mg
  ↓
5. 格式化后注入 prompt:
   ## 相关历史对话详情（从记忆中回溯）
   [历史对话详情 #1-#4]
   用户: 旺旺耳朵红红的怎么办
   助手: 建议使用耳肤灵滴耳液，每天两次
   ...
```

### Token 预算控制

- 默认预算：2000 tokens
- 递归深度上限：3 层
- 超出预算时截断消息并标注 `(... 更多历史消息已截断)`

### 文件

| 文件 | 说明 |
|------|------|
| `agent/memory_drilldown.py` | `MemoryDrillDown` 类（触发判断 + 递归钻取 + 格式化） |
| `agent/memory.py` | `SummaryDAG.drill_down()` / `find_relevant_summaries()` / `get_summary_by_id()` |
| `agent/chat.py` | `build_memory_context()` 中集成钻取逻辑 |

---

## 12. 模块 8：后台任务管理器（BackgroundTaskManager）

### 职责

对话后的非阻塞工作统一入口，使用 `ThreadPoolExecutor` 线程池。

### 背景

对话后需要执行偏好提取、DAG 压缩、标题生成等工作，这些都涉及 LLM API 调用（I/O 密集型），如果同步执行会让用户多等 1-3 秒。

### 架构

```
用户发消息 → LLM 回答 → 立即返回给用户
                ↓
        BackgroundTaskManager.submit()
                ↓
        ThreadPoolExecutor (max_workers=BG_MAX_WORKERS)
                ├─ 偏好提取: LLM 从对话中提取用户偏好 → 写入 user_preferences
                ├─ DAG 压缩: 检查未压缩消息是否超阈值 → 生成摘要
                └─ 标题生成: 前 3 轮后自动生成会话标题
```

### 配置

```bash
# .env
BG_MAX_WORKERS=4    # 后台线程数（默认 4，适合 10-50 并发用户）
```

| 用户规模 | 推荐 max_workers |
|---------|-----------------|
| 1-10 人 | 2 |
| 10-50 人 | 4（默认） |
| 50-200 人 | 8 |
| 200+ 人 | 考虑消息队列 |

### 队列保护

- 最大挂起任务数：100（超过则跳过，防止 LLM API 宕机时任务堆积）
- 自动清理已完成的 Future

### SQLite 线程安全

```python
sqlite3.connect(db_path, check_same_thread=False, timeout=10)
```

### LLM 自动偏好提取

每轮对话后，`BackgroundTaskManager` 自动提交偏好提取任务：

```
对话：
  用户: 我家狗叫旺旺，迷你雪纳瑞，3岁
  助手: 旺旺好可爱！...

后台：
  → build_extract_prompt() 构建提取 prompt
  → LLM 返回: {"new": ["用户的狗叫旺旺", "旺旺是迷你雪纳瑞", "旺旺3岁"], "update": []}
  → apply_extraction() 写入 user_preferences 表
```

### 文件

| 文件 | 说明 |
|------|------|
| `agent/background.py` | `BackgroundTaskManager` 类 + 3 个静态任务方法 |
| `agent/chat.py` | `chat_loop()` 中创建 bg_tasks，对话后 submit |
| `api/server.py` | `_bg_tasks` 全局单例，chat 端点中 submit |

---

## 13. 未来扩展

- **LLM 辅助钻取**：让 LLM 判断摘要是否足够回答问题，不足时再钻取（当前用关键词触发）
- **Checkpoint/Snapshot**：保存完整会话状态，支持回滚
- **多模态记忆**：支持图片/文件附件的记忆

---

## 14. 实施进度

- [x] 设计文档 v1（Conversation Store + Summary DAG + User Preferences + Query Rewrite）
- [x] 设计文档 v2.1（+ Embedding Memory + Session Management + 偏好级联更新）
- [x] 实现 memory.py（ConversationStore + UserPreferences + SummaryDAG）
- [x] 实现 compaction.py（DAG 压缩引擎）
- [x] 实现 memory_index.py（向量记忆检索）
- [x] 实现 session.py（会话管理）
- [x] 实现 query_rewrite.py（查询重写 + memory search）
- [x] 改造 chat.py 集成全部模块 + 多租户 + session
- [x] 测试：多租户隔离 + DAG 压缩 + 偏好级联更新链 + session 切换 + memory search（14/14 通过）
- [x] 实现 memory_drilldown.py（DAG 回溯钻取引擎）
- [x] SummaryDAG 新增 drill_down() / find_relevant_summaries() / get_summary_by_id()
- [x] chat.py 集成钻取到 build_memory_context()
- [x] 测试：DAG 回溯 + 钻取触发 + context 注入（test_memory 18/18 + test_e2e 26/26 通过）
- [x] 偏好 Decay：last_confirmed_at 字段 + _is_stale() + touch_preferences() + get_active_text() 过时标注
- [x] chat.py 集成偏好 decay（注入时标注 + 刷新确认时间）
- [x] 测试：偏好 decay（test_memory 23/23 + test_e2e 26/26 通过）
- [x] 后台任务管理器 background.py（ThreadPoolExecutor + 队列保护）
- [x] LLM 自动偏好提取（后台异步，不阻塞用户）
- [x] DAG 压缩 + 标题生成异步化
- [x] SQLite 线程安全（check_same_thread=False, timeout=10）
- [x] chat.py + server.py 集成 BackgroundTaskManager
- [x] .env.example 新增 BG_MAX_WORKERS
- [x] 测试通过（test_memory 23/23 + test_e2e 26/26）
