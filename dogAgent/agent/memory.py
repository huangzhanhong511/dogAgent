"""
dogAgent 记忆系统核心模块

包含三大核心组件：
- ConversationStore: 对话消息持久化（短期记忆）
- UserPreferences: 用户偏好提取与级联更新
- SummaryDAG: DAG 多层摘要压缩（长期记忆）

所有数据存储在 SQLite 中，按 user_id + session_id 隔离。
"""

import os
import uuid
import sqlite3
import logging
from datetime import datetime

logger = logging.getLogger("memory")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB_PATH = os.path.join(PROJECT_DIR, "data", "memory.db")


def _estimate_tokens(text: str) -> int:
    """粗略估算 token 数（中文约 1.5 字符/token，英文约 4 字符/token）"""
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other_chars = len(text) - chinese_chars
    return int(chinese_chars / 1.5 + other_chars / 4)


class MemoryDB:
    """SQLite 数据库管理器，所有记忆模块共享同一个连接"""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        if self.db_path != ":memory:":
            db_dir = os.path.dirname(self.db_path)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=10)
        self.conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self):
        """创建所有表结构"""
        cursor = self.conn.cursor()

        # 会话表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id         TEXT PRIMARY KEY,
                user_id    TEXT NOT NULL,
                title      TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active  BOOLEAN DEFAULT 1
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_user
            ON sessions(user_id, updated_at DESC)
        """)

        # 对话消息表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     TEXT NOT NULL,
                session_id  TEXT NOT NULL,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                token_count INTEGER,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_conv_user_session
            ON conversations(user_id, session_id, created_at)
        """)

        # 摘要节点表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                id              TEXT PRIMARY KEY,
                user_id         TEXT NOT NULL,
                session_id      TEXT NOT NULL,
                depth           INTEGER NOT NULL,
                content         TEXT NOT NULL,
                token_count     INTEGER,
                source_start_id INTEGER,
                source_end_id   INTEGER,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_summary_user_session
            ON summaries(user_id, session_id, depth)
        """)

        # DAG 边表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS summary_edges (
                parent_id  TEXT NOT NULL,
                child_id   TEXT NOT NULL,
                child_type TEXT NOT NULL,
                PRIMARY KEY (parent_id, child_id)
            )
        """)

        # 用户偏好表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id         TEXT NOT NULL,
                content         TEXT NOT NULL,
                source_msg_id   INTEGER,
                source_session_id TEXT,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                superseded_by   INTEGER DEFAULT NULL,
                last_confirmed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_pref_user_active
            ON user_preferences(user_id, superseded_by)
        """)

        # 兼容旧数据库：如果 last_confirmed_at 列不存在则添加
        try:
            cursor.execute("SELECT last_confirmed_at FROM user_preferences LIMIT 1")
        except sqlite3.OperationalError:
            cursor.execute(
                "ALTER TABLE user_preferences ADD COLUMN last_confirmed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            )

        # 用户认证表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS auth_users (
                id          TEXT PRIMARY KEY,
                username    TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                display_name TEXT,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_auth_username
            ON auth_users(username)
        """)

        # 当前有效视图表（LCM context_items）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS context_items (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     TEXT NOT NULL,
                session_id  TEXT NOT NULL,
                ordinal     INTEGER NOT NULL,
                item_type   TEXT NOT NULL CHECK (item_type IN ('message', 'summary')),
                message_id  INTEGER,
                summary_id  TEXT,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (user_id, session_id, ordinal)
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ctx_items_session
            ON context_items(user_id, session_id, ordinal)
        """)

        self.conn.commit()
        self._migrate_sessions_to_context_items(cursor)
        self.conn.commit()
        logger.info(f"记忆数据库已初始化: {self.db_path}")

    def _migrate_sessions_to_context_items(self, cursor):
        """将已有 session 的消息和摘要迁移到 context_items（幂等，仅处理尚未迁移的 session）"""
        cursor.execute("""
            SELECT DISTINCT user_id, session_id FROM conversations
            WHERE NOT EXISTS (
                SELECT 1 FROM context_items ci
                WHERE ci.user_id = conversations.user_id
                  AND ci.session_id = conversations.session_id
            )
        """)
        sessions = cursor.fetchall()
        for row in sessions:
            uid, sid = row["user_id"], row["session_id"]
            # 自由摘要：未被任何父 condensation 消费的摘要
            cursor.execute("""
                SELECT id, created_at FROM summaries
                WHERE user_id = ? AND session_id = ?
                  AND id NOT IN (
                      SELECT child_id FROM summary_edges WHERE child_type = 'summary'
                  )
                ORDER BY created_at
            """, (uid, sid))
            free_summaries = cursor.fetchall()

            # 未压缩消息：id > 所有摘要的 source_end_id
            cursor.execute("""
                SELECT COALESCE(MAX(source_end_id), 0) as max_id
                FROM summaries WHERE user_id = ? AND session_id = ?
            """, (uid, sid))
            max_compressed = cursor.fetchone()["max_id"]

            cursor.execute("""
                SELECT id FROM conversations
                WHERE user_id = ? AND session_id = ? AND id > ?
                ORDER BY created_at
            """, (uid, sid, max_compressed))
            unconsumed = cursor.fetchall()

            ordinal = 0
            for s in free_summaries:
                cursor.execute(
                    "INSERT OR IGNORE INTO context_items (user_id, session_id, ordinal, item_type, summary_id) VALUES (?, ?, ?, 'summary', ?)",
                    (uid, sid, ordinal, s["id"]),
                )
                ordinal += 1
            for m in unconsumed:
                cursor.execute(
                    "INSERT OR IGNORE INTO context_items (user_id, session_id, ordinal, item_type, message_id) VALUES (?, ?, ?, 'message', ?)",
                    (uid, sid, ordinal, m["id"]),
                )
                ordinal += 1
            if ordinal > 0:
                logger.info(f"迁移 session {sid}: {ordinal} 条 context_items")

    def close(self):
        if self.conn:
            self.conn.close()


class ConversationStore:
    """对话消息持久化（短期记忆）"""

    def __init__(self, db: MemoryDB):
        self.db = db

    def add_message(self, user_id: str, session_id: str, role: str, content: str) -> int:
        """存储一条消息，返回消息 ID，同时追加到 context_items"""
        token_count = _estimate_tokens(content)
        cursor = self.db.conn.cursor()
        cursor.execute(
            "INSERT INTO conversations (user_id, session_id, role, content, token_count) VALUES (?, ?, ?, ?, ?)",
            (user_id, session_id, role, content, token_count),
        )
        msg_id = cursor.lastrowid
        cursor.execute(
            "SELECT COALESCE(MAX(ordinal), -1) AS max_ord FROM context_items WHERE user_id = ? AND session_id = ?",
            (user_id, session_id),
        )
        max_ord = cursor.fetchone()["max_ord"]
        cursor.execute(
            "INSERT INTO context_items (user_id, session_id, ordinal, item_type, message_id) VALUES (?, ?, ?, 'message', ?)",
            (user_id, session_id, max_ord + 1, msg_id),
        )
        self.db.conn.commit()
        logger.debug(f"消息已存储: id={msg_id}, role={role}, tokens={token_count}")
        return msg_id

    def get_recent(self, user_id: str, session_id: str, limit: int = 20) -> list[dict]:
        """获取当前 session 最近 N 条消息"""
        cursor = self.db.conn.cursor()
        cursor.execute(
            """SELECT id, role, content, token_count, created_at
               FROM conversations
               WHERE user_id = ? AND session_id = ?
               ORDER BY created_at DESC
               LIMIT ?""",
            (user_id, session_id, limit),
        )
        rows = cursor.fetchall()
        # 返回时按时间正序
        return [dict(r) for r in reversed(rows)]

    def get_range(self, user_id: str, start_id: int, end_id: int) -> list[dict]:
        """获取指定 ID 范围的消息（用于 DAG 回溯）"""
        cursor = self.db.conn.cursor()
        cursor.execute(
            """SELECT id, role, content, token_count, created_at
               FROM conversations
               WHERE user_id = ? AND id >= ? AND id <= ?
               ORDER BY created_at""",
            (user_id, start_id, end_id),
        )
        return [dict(r) for r in cursor.fetchall()]


class UserPreferences:
    """用户偏好管理（自然语言存储 + 级联更新）"""

    EXTRACT_PROMPT = """从以下对话中提取用户关于其宠物或个人偏好的信息。
用自然语言描述，每条一行。

规则：
1. 提取新出现的信息（关于宠物的名字、品种、年龄、体重、健康状况、饮食、行为等）
2. 如果用户更新了已有信息（如改名、年龄变化），标记为 update
3. 级联更新：如果核心属性变化（如狗的名字），所有引用该属性的偏好都要更新
4. 如果没有新信息，返回空

现有偏好：
{existing_preferences}

最新对话：
{conversation}

请返回 JSON：
{{
  "new": ["新偏好1", "新偏好2"],
  "update": [
    {{"old": "旧偏好内容", "new": "新偏好内容"}}
  ]
}}
如果没有新信息，返回 {{"new": [], "update": []}}"""

    def __init__(self, db: MemoryDB):
        self.db = db

    def get_active(self, user_id: str) -> list[dict]:
        """获取用户所有活跃偏好"""
        cursor = self.db.conn.cursor()
        cursor.execute(
            """SELECT id, content, source_msg_id, source_session_id, created_at, last_confirmed_at
               FROM user_preferences
               WHERE user_id = ? AND superseded_by IS NULL
               ORDER BY created_at""",
            (user_id,),
        )
        return [dict(r) for r in cursor.fetchall()]

    # 超过此天数未确认的偏好视为"过时"
    STALE_DAYS = 30

    def get_active_text(self, user_id: str) -> str:
        """获取用户偏好的纯文本（用于注入 prompt），过时偏好加标注"""
        prefs = self.get_active(user_id)
        if not prefs:
            return ""
        lines = []
        for p in prefs:
            text = p["content"]
            if self._is_stale(p):
                text += "  (较久前的信息，可能已过时)"
            lines.append(f"- {text}")
        return "关于该用户已知信息：\n" + "\n".join(lines)

    def _is_stale(self, pref: dict) -> bool:
        """判断偏好是否已过时（超过 STALE_DAYS 天未确认）"""
        confirmed = pref.get("last_confirmed_at")
        if not confirmed:
            return False
        try:
            if isinstance(confirmed, str):
                confirmed_dt = datetime.fromisoformat(confirmed.replace("Z", "+00:00"))
            else:
                confirmed_dt = confirmed
            age_days = (datetime.now() - confirmed_dt.replace(tzinfo=None)).days
            return age_days > self.STALE_DAYS
        except (ValueError, TypeError):
            return False

    def touch_preferences(self, user_id: str, pref_ids: list[int] = None):
        """
        更新偏好的 last_confirmed_at（表示本轮对话"用到了"这些偏好）。

        Args:
            user_id: 用户 ID
            pref_ids: 要更新的偏好 ID 列表，为 None 则更新所有活跃偏好
        """
        cursor = self.db.conn.cursor()
        if pref_ids:
            placeholders = ",".join("?" * len(pref_ids))
            cursor.execute(
                f"""UPDATE user_preferences
                    SET last_confirmed_at = CURRENT_TIMESTAMP
                    WHERE user_id = ? AND id IN ({placeholders}) AND superseded_by IS NULL""",
                [user_id] + pref_ids,
            )
        else:
            cursor.execute(
                """UPDATE user_preferences
                   SET last_confirmed_at = CURRENT_TIMESTAMP
                   WHERE user_id = ? AND superseded_by IS NULL""",
                (user_id,),
            )
        self.db.conn.commit()
        logger.debug(f"偏好已确认: user={user_id}, ids={pref_ids or 'all'}")

    def add_preference(
        self, user_id: str, content: str,
        source_msg_id: int = None, source_session_id: str = None
    ) -> int:
        """添加新偏好，返回 ID"""
        cursor = self.db.conn.cursor()
        cursor.execute(
            """INSERT INTO user_preferences (user_id, content, source_msg_id, source_session_id)
               VALUES (?, ?, ?, ?)""",
            (user_id, content, source_msg_id, source_session_id),
        )
        self.db.conn.commit()
        new_id = cursor.lastrowid
        logger.info(f"新偏好: id={new_id}, content={content[:50]}")
        return new_id

    def update_preference(self, old_id: int, new_content: str, user_id: str,
                          source_msg_id: int = None, source_session_id: str = None) -> int:
        """更新偏好（标记旧的为 superseded，插入新的）"""
        cursor = self.db.conn.cursor()
        # 插入新偏好
        cursor.execute(
            """INSERT INTO user_preferences (user_id, content, source_msg_id, source_session_id)
               VALUES (?, ?, ?, ?)""",
            (user_id, new_content, source_msg_id, source_session_id),
        )
        new_id = cursor.lastrowid
        # 标记旧偏好
        cursor.execute(
            "UPDATE user_preferences SET superseded_by = ? WHERE id = ?",
            (new_id, old_id),
        )
        self.db.conn.commit()
        logger.info(f"偏好更新: {old_id} → {new_id}, content={new_content[:50]}")
        return new_id

    def apply_extraction(self, user_id: str, extraction: dict,
                         source_msg_id: int = None, source_session_id: str = None):
        """应用 LLM 提取结果（new + update）"""
        # 处理新增
        for content in extraction.get("new", []):
            self.add_preference(user_id, content, source_msg_id, source_session_id)

        # 处理更新（级联）
        active_prefs = self.get_active(user_id)
        for upd in extraction.get("update", []):
            old_content = upd.get("old", "")
            new_content = upd.get("new", "")
            # 找到匹配的旧偏好
            matched = None
            for p in active_prefs:
                if p["content"] == old_content:
                    matched = p
                    break
            if matched:
                self.update_preference(
                    matched["id"], new_content, user_id,
                    source_msg_id, source_session_id
                )
            else:
                # 旧偏好未精确匹配，作为新偏好插入
                logger.warning(f"偏好未匹配到旧记录，作为新增: {new_content[:50]}")
                self.add_preference(user_id, new_content, source_msg_id, source_session_id)

    def build_extract_prompt(self, user_id: str, conversation_text: str) -> str:
        """构建偏好提取的 LLM prompt"""
        active_prefs = self.get_active(user_id)
        existing = "\n".join(f"- {p['content']}" for p in active_prefs) if active_prefs else "(无)"
        return self.EXTRACT_PROMPT.format(
            existing_preferences=existing,
            conversation=conversation_text,
        )


class SummaryDAG:
    """DAG 多层摘要管理（长期记忆）"""

    def __init__(self, db: MemoryDB):
        self.db = db

    def add_summary(
        self, user_id: str, session_id: str, depth: int, content: str,
        source_start_id: int = None, source_end_id: int = None,
        child_ids: list[str] = None, child_types: list[str] = None,
    ) -> str:
        """添加摘要节点并建立 DAG 边"""
        summary_id = str(uuid.uuid4())
        token_count = _estimate_tokens(content)
        cursor = self.db.conn.cursor()

        cursor.execute(
            """INSERT INTO summaries (id, user_id, session_id, depth, content, token_count,
               source_start_id, source_end_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (summary_id, user_id, session_id, depth, content, token_count,
             source_start_id, source_end_id),
        )

        # 建立 DAG 边
        if child_ids and child_types:
            for child_id, child_type in zip(child_ids, child_types):
                cursor.execute(
                    "INSERT INTO summary_edges (parent_id, child_id, child_type) VALUES (?, ?, ?)",
                    (summary_id, child_id, child_type),
                )

        self.db.conn.commit()
        logger.info(f"摘要已存储: id={summary_id[:8]}, depth={depth}, tokens={token_count}")

        return summary_id

    def append_context_summary(self, user_id: str, session_id: str, summary_id: str) -> None:
        """将摘要追加到 context_items 末尾（用于测试辅助和手动场景）。"""
        cursor = self.db.conn.cursor()
        cursor.execute(
            "SELECT COALESCE(MAX(ordinal), -1) + 1 FROM context_items WHERE user_id = ? AND session_id = ?",
            (user_id, session_id),
        )
        next_ord = cursor.fetchone()[0]
        cursor.execute(
            "INSERT INTO context_items (user_id, session_id, ordinal, item_type, summary_id) VALUES (?, ?, ?, 'summary', ?)",
            (user_id, session_id, next_ord, summary_id),
        )
        self.db.conn.commit()

    # ─── context_items 管理（LCM 当前视图） ───

    def replace_context_range(
        self, user_id: str, session_id: str,
        start_ordinal: int, end_ordinal: int, summary_id: str,
    ) -> None:
        """
        将 context_items 中 [start_ordinal, end_ordinal] 范围替换为一个 summary 条目。
        严格按照 LCM replaceContextRangeWithSummary 3步原子操作：
          1. 删除范围内所有条目
          2. 在 start_ordinal 插入新 summary
          3. 双遍重排序（用负数临时值避免 UNIQUE 冲突）
        """
        cursor = self.db.conn.cursor()
        cursor.execute(
            "DELETE FROM context_items WHERE user_id = ? AND session_id = ? AND ordinal >= ? AND ordinal <= ?",
            (user_id, session_id, start_ordinal, end_ordinal),
        )
        cursor.execute(
            "INSERT INTO context_items (user_id, session_id, ordinal, item_type, summary_id) VALUES (?, ?, ?, 'summary', ?)",
            (user_id, session_id, start_ordinal, summary_id),
        )
        cursor.execute(
            "SELECT ordinal FROM context_items WHERE user_id = ? AND session_id = ? ORDER BY ordinal",
            (user_id, session_id),
        )
        items = cursor.fetchall()
        if items and any(row["ordinal"] != i for i, row in enumerate(items)):
            for i, row in enumerate(items):
                cursor.execute(
                    "UPDATE context_items SET ordinal = ? WHERE user_id = ? AND session_id = ? AND ordinal = ?",
                    (-(i + 1), user_id, session_id, row["ordinal"]),
                )
            for i in range(len(items)):
                cursor.execute(
                    "UPDATE context_items SET ordinal = ? WHERE user_id = ? AND session_id = ? AND ordinal = ?",
                    (i, user_id, session_id, -(i + 1)),
                )
        self.db.conn.commit()

    def get_context_items(self, user_id: str, session_id: str) -> list[dict]:
        """按 ordinal 顺序返回所有 context_items（含 item_type / message_id / summary_id）"""
        cursor = self.db.conn.cursor()
        cursor.execute(
            """SELECT ordinal, item_type, message_id, summary_id
               FROM context_items WHERE user_id = ? AND session_id = ? ORDER BY ordinal""",
            (user_id, session_id),
        )
        return [dict(r) for r in cursor.fetchall()]

    def get_context_messages(self, user_id: str, session_id: str) -> list[dict]:
        """返回 context_items 中 message 条目的完整内容，按 ordinal 顺序（即 fresh tail）。"""
        cursor = self.db.conn.cursor()
        cursor.execute(
            """SELECT c.id, c.role, c.content
               FROM context_items ci
               JOIN conversations c ON c.id = ci.message_id
               WHERE ci.user_id = ? AND ci.session_id = ? AND ci.item_type = 'message'
               ORDER BY ci.ordinal""",
            (user_id, session_id),
        )
        return [dict(r) for r in cursor.fetchall()]

    def get_context_token_count(self, user_id: str, session_id: str) -> int:
        """计算 context_items 中所有条目的 token 总量（消息 + 摘要）"""
        cursor = self.db.conn.cursor()
        cursor.execute("""
            SELECT COALESCE(SUM(token_count), 0) AS total FROM (
                SELECT c.token_count
                FROM context_items ci
                JOIN conversations c ON c.id = ci.message_id
                WHERE ci.user_id = ? AND ci.session_id = ? AND ci.item_type = 'message'

                UNION ALL

                SELECT s.token_count
                FROM context_items ci
                JOIN summaries s ON s.id = ci.summary_id
                WHERE ci.user_id = ? AND ci.session_id = ? AND ci.item_type = 'summary'
            ) sub
        """, (user_id, session_id, user_id, session_id))
        return cursor.fetchone()["total"]

    def get_distinct_depths_in_context(self, user_id: str, session_id: str) -> list[int]:
        """返回 context_items 中所有 summary 的 depth 列表（升序，无重复）"""
        cursor = self.db.conn.cursor()
        cursor.execute("""
            SELECT DISTINCT s.depth
            FROM context_items ci
            JOIN summaries s ON s.id = ci.summary_id
            WHERE ci.user_id = ? AND ci.session_id = ? AND ci.item_type = 'summary'
            ORDER BY s.depth ASC
        """, (user_id, session_id))
        return [r["depth"] for r in cursor.fetchall()]

    def get_children(self, summary_id: str) -> list[dict]:
        """获取摘要的子节点（用于 DAG 回溯）"""
        cursor = self.db.conn.cursor()
        cursor.execute(
            """SELECT child_id, child_type FROM summary_edges
               WHERE parent_id = ?""",
            (summary_id,),
        )
        return [dict(r) for r in cursor.fetchall()]

    def get_context_text(self, user_id: str, session_id: str) -> str:
        """获取用于 prompt 注入的摘要 XML（从 context_items 读取，按 ordinal 顺序混合深度）。"""
        cursor = self.db.conn.cursor()
        cursor.execute("""
            SELECT ci.ordinal, s.id, s.depth, s.content, s.created_at
            FROM context_items ci
            JOIN summaries s ON s.id = ci.summary_id
            WHERE ci.user_id = ? AND ci.session_id = ? AND ci.item_type = 'summary'
            ORDER BY ci.ordinal
        """, (user_id, session_id))
        rows = cursor.fetchall()

        if not rows:
            return ""

        parts = []
        for r in rows:
            created = (r["created_at"] or "")[:16].replace("T", " ")
            kind = "leaf" if r["depth"] == 0 else "condensed"
            attrs = f'id="{r["id"]}" depth="{r["depth"]}" kind="{kind}"'
            if created:
                attrs += f' created="{created}"'
            parts.append(f'<summary {attrs}>\n{r["content"]}\n</summary>')

        return "\n\n".join(parts)

    # ─── DAG 回溯（Lossless Claw drill-down） ───

    def get_summary_by_id(self, summary_id: str) -> dict | None:
        """根据 ID 获取单个摘要"""
        cursor = self.db.conn.cursor()
        cursor.execute(
            """SELECT id, user_id, session_id, depth, content, token_count,
                      source_start_id, source_end_id, created_at
               FROM summaries WHERE id = ?""",
            (summary_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def drill_down(self, summary_id: str, user_id: str,
                   conv_store: 'ConversationStore') -> list[dict]:
        """
        从指定摘要向下钻取一层（Lossless Claw 回溯）。

        返回子节点列表，每个元素为：
          {"type": "summary", "id": ..., "content": ..., "depth": ...}
          或
          {"type": "messages", "messages": [...], "start_id": ..., "end_id": ...}
        """
        children = self.get_children(summary_id)
        if not children:
            summary = self.get_summary_by_id(summary_id)
            if summary and summary.get("source_start_id") and summary.get("source_end_id"):
                messages = conv_store.get_range(
                    user_id, summary["source_start_id"], summary["source_end_id"]
                )
                if messages:
                    return [{
                        "type": "messages",
                        "messages": messages,
                        "start_id": summary["source_start_id"],
                        "end_id": summary["source_end_id"],
                    }]
            return []

        results = []
        for child in children:
            if child["child_type"] == "summary":
                child_summary = self.get_summary_by_id(child["child_id"])
                if child_summary:
                    results.append({
                        "type": "summary",
                        "id": child_summary["id"],
                        "content": child_summary["content"],
                        "depth": child_summary["depth"],
                    })
            elif child["child_type"] == "message_range":
                range_str = child["child_id"]
                if ":" in range_str:
                    _, range_part = range_str.split(":", 1)
                    if "-" in range_part:
                        start_str, end_str = range_part.split("-", 1)
                        try:
                            start_id = int(start_str)
                            end_id = int(end_str)
                            messages = conv_store.get_range(user_id, start_id, end_id)
                            if messages:
                                results.append({
                                    "type": "messages",
                                    "messages": messages,
                                    "start_id": start_id,
                                    "end_id": end_id,
                                })
                        except ValueError:
                            logger.warning(f"无法解析消息范围: {range_str}")
        return results

