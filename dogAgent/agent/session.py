"""
dogAgent 会话管理模块

管理用户的多个对话会话，区分不同主题。
每个 session 拥有独立的对话历史和 DAG 摘要树。
User Preferences 跨 session 共享。
"""

import uuid
import logging
from datetime import datetime

logger = logging.getLogger("session")


class SessionManager:
    """多会话管理器"""

    TITLE_PROMPT = """根据以下对话，生成一个简短的会话标题（10字以内）：

{conversation}

标题："""

    def __init__(self, db):
        """
        Args:
            db: MemoryDB 实例
        """
        self.db = db

    def create_session(self, user_id: str, title: str = None) -> str:
        """创建新会话，返回 session_id"""
        session_id = str(uuid.uuid4())
        cursor = self.db.conn.cursor()
        cursor.execute(
            "INSERT INTO sessions (id, user_id, title) VALUES (?, ?, ?)",
            (session_id, user_id, title),
        )
        self.db.conn.commit()
        logger.info(f"新会话: id={session_id[:8]}, user={user_id}, title={title}")
        return session_id

    def get_or_create_session(self, user_id: str) -> str:
        """获取最近活跃的 session，如果没有则创建新的"""
        session = self.get_latest_session(user_id)
        if session:
            return session["id"]
        return self.create_session(user_id)

    def get_latest_session(self, user_id: str) -> dict | None:
        """获取用户最近活跃的 session"""
        cursor = self.db.conn.cursor()
        cursor.execute(
            """SELECT id, title, created_at, updated_at, is_active
               FROM sessions
               WHERE user_id = ? AND is_active = 1
               ORDER BY updated_at DESC
               LIMIT 1""",
            (user_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_session(self, session_id: str) -> dict | None:
        """根据 ID 获取 session"""
        cursor = self.db.conn.cursor()
        cursor.execute(
            "SELECT id, user_id, title, created_at, updated_at, is_active FROM sessions WHERE id = ?",
            (session_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def list_sessions(self, user_id: str, include_archived: bool = False) -> list[dict]:
        """列出用户所有 session"""
        cursor = self.db.conn.cursor()
        if include_archived:
            cursor.execute(
                """SELECT id, title, created_at, updated_at, is_active
                   FROM sessions
                   WHERE user_id = ?
                   ORDER BY updated_at DESC""",
                (user_id,),
            )
        else:
            cursor.execute(
                """SELECT id, title, created_at, updated_at, is_active
                   FROM sessions
                   WHERE user_id = ? AND is_active = 1
                   ORDER BY updated_at DESC""",
                (user_id,),
            )
        return [dict(r) for r in cursor.fetchall()]

    def touch_session(self, session_id: str):
        """更新 session 的 updated_at（每次对话时调用）"""
        cursor = self.db.conn.cursor()
        cursor.execute(
            "UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (session_id,),
        )
        self.db.conn.commit()

    def update_title(self, session_id: str, title: str):
        """更新会话标题"""
        cursor = self.db.conn.cursor()
        cursor.execute(
            "UPDATE sessions SET title = ? WHERE id = ?",
            (title, session_id),
        )
        self.db.conn.commit()
        logger.info(f"会话标题更新: {session_id[:8]} → {title}")

    def archive_session(self, session_id: str):
        """归档会话"""
        cursor = self.db.conn.cursor()
        cursor.execute(
            "UPDATE sessions SET is_active = 0 WHERE id = ?",
            (session_id,),
        )
        self.db.conn.commit()
        logger.info(f"会话已归档: {session_id[:8]}")

    def get_message_count(self, session_id: str) -> int:
        """获取会话的消息数量"""
        cursor = self.db.conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) as cnt FROM conversations WHERE session_id = ?",
            (session_id,),
        )
        return cursor.fetchone()["cnt"]

    def needs_title(self, session_id: str) -> bool:
        """判断会话是否需要自动生成标题（无标题且消息 >= 3 轮）"""
        session = self.get_session(session_id)
        if not session or session["title"]:
            return False
        msg_count = self.get_message_count(session_id)
        return msg_count >= 6  # 3 轮 = 6 条消息 (user + assistant)

    def build_title_prompt(self, session_id: str) -> str:
        """构建标题生成的 LLM prompt"""
        cursor = self.db.conn.cursor()
        cursor.execute(
            """SELECT role, content FROM conversations
               WHERE session_id = ?
               ORDER BY created_at
               LIMIT 6""",
            (session_id,),
        )
        rows = cursor.fetchall()
        lines = []
        for r in rows:
            prefix = "用户" if r["role"] == "user" else "助手"
            # 截断过长的内容
            content = r["content"][:200]
            lines.append(f"{prefix}：{content}")

        conversation = "\n".join(lines)
        return self.TITLE_PROMPT.format(conversation=conversation)

    def ensure_session(self, user_id: str, session_id: str = None) -> str:
        """
        确保会话存在并返回 session_id。

        Args:
            user_id: 用户 ID
            session_id: 指定的 session_id，为 None 则获取最近的或创建新的
                       为 "new" 则创建新会话

        Returns:
            有效的 session_id
        """
        if session_id == "new":
            return self.create_session(user_id)

        if session_id:
            session = self.get_session(session_id)
            if session:
                return session_id
            else:
                logger.warning(f"Session {session_id} 不存在，创建新会话")
                return self.create_session(user_id)

        return self.get_or_create_session(user_id)