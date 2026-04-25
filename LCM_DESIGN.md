# LCM 设计文档

Lossless Claw Memory 的核心思路：用一张有序的 `context_items` 表作为"当前有效视图"，
压缩 = 原地替换，原始数据永不删除，drill-down 随时可还原。

---

## 核心数据结构

```
context_items          ← 当前有效视图（平铺有序列表）
  conversation_id
  ordinal              ← 严格连续 0,1,2,3...（无空洞）
  item_type            ← 'message' | 'summary'
  message_id           ← item_type='message' 时有值
  summary_id           ← item_type='summary' 时有值

summaries              ← 所有摘要档案（永久保留，从不删除）
  id
  depth                ← 0=leaf, 1=condensed, 2+=higher
  kind                 ← 'leaf' | 'condensed'
  content
  token_count

messages               ← 原始消息（永久保留）
  id, role, content

summary_messages       ← leaf summary → 原始消息（多对多）
  summary_id, message_id

summary_parents        ← condensed summary → 子 summary（多对多）
  parent_id, child_id
```

---

## 关键参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `fresh_tail_count` | 8 | 保护最新 N 条**原始消息**不被压缩 |
| `leaf_min_fanout` | 8 | fresh tail 外积累到 N 条原始消息时触发 Leaf 压缩 |
| `condensed_min_fanout` | 6 | 同层积累到 N 个 leaf summary 时触发 Condensation |

---

## 演进流程

### 阶段一：纯原始消息

```
context_items:
  0: msg#1
  1: msg#2
  ...
  15: msg#16

fresh tail = ordinals 8-15（最新8条原始消息，受保护）
compactable zone = ordinals 0-7（8条，达到 leaf_min_fanout）
```

---

### 阶段二：Leaf 压缩触发

**条件**：fresh tail 之外有 >= `leaf_min_fanout` 条原始消息

**操作**：删除 ordinals 0-7，插入1个 leaf summary，重新编号

```
原地替换：
  [msg#1][msg#2]...[msg#8]  →  [leaf_001]
                                 ↑ depth=0，覆盖 msg#1-8

context_items 之后：
  0: [leaf_001]    ← 替换了8条原始消息
  1: msg#9
  2: msg#10
  ...
  8: msg#16        ← fresh tail 仍然是最新8条原始消息

LLM 看到: [leaf_001] + msg#9-16
```

*summaries 表、messages 表不变，summary_messages 记录 leaf_001 → msg#1-8*

---

### 阶段三：Leaf 不断积累

每当 fresh tail 外积累到 8 条原始消息，再次触发 Leaf 压缩：

```
经过4轮压缩后：

context_items:
  0: [leaf_001]   depth=0, 覆盖 msg#1-8
  1: [leaf_002]   depth=0, 覆盖 msg#9-16
  2: [leaf_003]   depth=0, 覆盖 msg#17-24
  3: [leaf_004]   depth=0, 覆盖 msg#25-32
  4: msg#33  ←── fresh tail
  ...
  11: msg#40

LLM 看到: [leaf_001][leaf_002][leaf_003][leaf_004] + msg#33-40
```

**关键**：context_items 里同时有多个 depth=0 的 leaf summary，
LLM 全都能看到，不只是"最高层"。

---

### 阶段四：Condensation 触发

**条件**：context_items 里 depth=0 的 leaf summary 数量 >= `condensed_min_fanout`

**操作**：删除这批 leaf summaries，插入1个 condensed summary

```
积累到6个 leaf summaries 后：

context_items（触发前）：
  0: [leaf_001]   depth=0
  1: [leaf_002]   depth=0
  2: [leaf_003]   depth=0
  3: [leaf_004]   depth=0
  4: [leaf_005]   depth=0
  5: [leaf_006]   depth=0
  6: msg#...  ←── fresh tail
  ...

原地替换 ordinals 0-5：
  [leaf_001..006]  →  [condensed_001]  depth=1

context_items（触发后）：
  0: [condensed_001]  depth=1，覆盖所有旧 leaf
  1: msg#...  ←── fresh tail
  ...

LLM 看到: [condensed_001] + 最新8条原始消息
```

*leaf_001..006 仍在 summaries 表，summary_parents 记录 condensed_001 → leaf_001-006*

---

### 阶段五：混合层级（正常运行态）

Condensation 之后继续积累，context_items 会同时出现不同 depth 的 summary：

```
context_items:
  0: [condensed_001]   depth=1   ← 老历史
  1: [leaf_007]        depth=0   ← 新积累的
  2: [leaf_008]        depth=0
  3: msg#...  ←── fresh tail
  ...

LLM 看到: [condensed_001] + [leaf_007][leaf_008] + 最新8条原始消息
          ↑ depth=1          ↑ depth=0              ↑ 原始消息
          混合展示，按时间顺序排列
```

这是 **LCM 能自然展示混合 depth** 的原因：
context_items 是平铺列表，深度不影响显示，只影响是否触发下一轮 condensation。

---

## Drill-Down 机制（无损还原）

LLM 看到 condensed_001 的 XML：

```xml
<summary id="condensed_001" depth="1">
  狗狗有耳炎，进行了药物治疗，饮食有所调整...
  Expand for details about: 耳肤灵品牌、具体剂量、换粮时间表
</summary>
```

LLM 判断需要细节 → 调用 `memory_expand("condensed_001")`：

```
condensed_001
  ↓ summary_parents
  leaf_001  leaf_002  leaf_003  ...  leaf_006
  （LLM 看到6个 leaf 摘要，可继续展开任意一个）

leaf_001
  ↓ summary_messages
  msg#1（verbatim）  msg#2（verbatim）  ...  msg#8（verbatim）
  （原始对话，一字不差）
```

**"无损"的含义**：
- 数据层：原始消息永远在 messages 表，随时可还原 ✅
- 语义层：能否找到取决于摘要的 `Expand for details about:` 质量 ⚠️

---