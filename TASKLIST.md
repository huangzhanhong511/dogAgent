# dogAgent 记忆系统改善任务列表

## 架构方向（2026-04-23 确定）

```
跨 Session   → user_preferences 文本 blob，LLM 提取，直接注入 prompt  ✅ 不动
Session 内   → DAG 压缩 + LLM 驱动展开（对齐 Lossless Claw）          ← 核心方向
领域知识     → Wiki RAG                                               ✅ 不动
```

**核心原则：把"什么信息相关"的判断交给 LLM，代码只负责压缩和存取。**

---

## 进度总览

| # | 任务 | 状态 | 优先级 |
|---|------|------|--------|
| 1 | Session 内检索改为 LLM 驱动（XML 摘要 + tool） | ✅ 完成 | P1 |
| 2 | FTS5 替换 memory_index 向量全扫 | ⬜ 待开始 | P1 |
| 3 | Condensation 策略从全量合并改为滑动窗口 | ⬜ 待开始 | P2 |
| 4 | Token 估算从启发式函数替换为 tiktoken 精确计算 | ⬜ 待开始 | P3 |

### 已完成
| # | 任务 | 完成日期 |
|---|------|----------|
| ✅ | P1-A Session 内检索改为 LLM 驱动 | 2026-04-23 |
| ✅ | _drill_recursive 每层向量选最相关子节点 | 2026-04-23 |

### 已废弃（架构讨论后取消）
| # | 任务 | 原因 |
|---|------|------|
| ~~跨 session DAG 压缩~~ | 只有特征信息需要跨 session，普通对话内容不需要 |
| ~~memory_index 跨 session 向量检索~~ | 过度设计，user_preferences 直接注入已够用 |
| ~~drill-down 关键词触发改语义分类器~~ | 被 P1 LLM 驱动方案整体替代 |
| ~~摘要向量化粒度问题~~ | LLM 驱动方案不依赖向量匹配，问题消失 |

---

## 任务详情

### P1-A · Session 内检索改为 LLM 驱动

**背景**
原始 Lossless Claw 的做法：摘要以 XML 格式直接在 prompt 里，
LLM 自己判断是否需要展开，调用 tool 后由代码取原始消息。
和现有 LLMWikiIndexRetriever 哲学完全一致。

**问题**
现有代码用关键词/向量启发式触发 drill-down，
代码的语义理解能力永远不如 LLM，且维护成本高。

**目标**
1. 顶层摘要改为 XML 格式，带时间范围和 "Expand for details about: ..." 提示
2. 实现 `memory_expand` tool，LLM 调用后触发 drill_down
3. 删除 `needs_drilldown()` 关键词逻辑
4. `build_memory_context()` 只注入 XML 摘要，不再主动触发钻取

**涉及文件**
- `agent/memory.py` — `SummaryDAG.get_context_text()` 改为 XML 输出
- `agent/memory_drilldown.py` — 删除 `needs_drilldown()`，添加 `drilldown_by_id()` 作为 tool 实现
- `agent/chat.py` — `build_memory_context()` 简化；新增 `create_memory_expand_tool()` 和 `invoke_with_tools()`
- `api/server.py` — chat 端点改用 `invoke_with_tools`

**状态**: ✅ 完成（2026-04-23）

---

### P1-B · FTS5 替换 memory_index 向量全扫

**背景**
原始 Lossless Claw 用 FTS5 全文检索（SQLite 内置），
不用 embedding，对专有名词（"耳肤灵"、"胰腺炎"）反而更精准。

**问题**
现有 `memory_index.py` 对每条记录做 Python cosine 计算，
数据量增大后线性增长；embedding API 有额外成本。

**目标**
在 `summaries` 表上建 FTS5 虚拟表，
`find_relevant_summaries` 降级 fallback 改为 FTS5 查询。
（向量检索作为可选增强保留，不删除）

**涉及文件**
- `agent/memory.py` — `MemoryDB._init_tables()` 加 FTS5 表；`find_relevant_summaries()` fallback 改 FTS5
- `agent/compaction.py` — `add_summary()` 后同步写 FTS5

**状态**: ⬜ 待开始

---

### P2 · Condensation 策略改为滑动窗口

**问题**
`compaction.py` 的 `_condense_level()` 把同层全部摘要一次性合并，
导致最新的叶子摘要也被过早压缩，损失近期细节。

**目标**
只合并最老的 N 个摘要（sliding window），
保留最近 `min_fanout` 个叶子摘要不参与本轮 condensation。

**涉及文件**
- `agent/compaction.py` — `_condense_level()`

**状态**: ⬜ 待开始

---

### P3 · Token 估算换 tiktoken 精确计算

**问题**
多处使用启发式估算 `cn / 1.5 + others / 4`，
中文文本误差可达 20-30%，影响压缩触发时机。

**目标**
引入 `tiktoken`，按实际使用的模型精确计算 token 数，
无 tiktoken 时自动降级启发式。

**涉及文件**
- `agent/memory.py` — `_estimate_tokens()`
- `agent/memory_drilldown.py` — `_estimate_tokens()`
- `agent/compaction.py` — 所有调用处

**状态**: ⬜ 待开始

---

## 变更记录

| 日期 | 任务 | 变更内容 |
|------|------|----------|
| 2026-04-23 | P0 find_relevant_summaries | `SummaryDAG` 接入 `MemoryIndex`；`add_summary` 自动写入 embedding；`find_relevant_summaries` 优先向量检索、降级关键词；`create_embedder()` 新增；chat.py / server.py 完成接线 |
| 2026-04-23 | P0 修复 DAG 层级查找 | `find_relevant_summaries` 改为只在顶层节点匹配（不再全表扫）；`_drill_recursive` 每层多个子节点时用向量选最相关分支；`MemoryDrillDown` 接入 `memory_index` |
| 2026-04-23 | P1-A Session 内检索 LLM 驱动 | XML 摘要注入；`memory_expand` tool；删除 `needs_drilldown()`；`build_memory_context` 去掉跨 session 向量检索和关键词触发 |
