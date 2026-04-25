"""
dogAgent 用户认证模块

功能：
- 用户注册 / 登录
- JWT token 签发 / 校验
- FastAPI 依赖注入 get_current_user
"""

import os
import uuid
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
from jose import JWTError, jwt

logger = logging.getLogger("auth")

# === 配置 ===
JWT_SECRET = os.getenv("JWT_SECRET", "dogagent-dev-secret-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_DAYS = 7


def _hash_password(password: str) -> str:
    """bcrypt 哈希密码"""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _verify_password(password: str, hashed: str) -> bool:
    """验证密码"""
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))


class AuthService:
    """用户认证服务"""

    def __init__(self, db):
        """
        Args:
            db: MemoryDB 实例（包含 auth_users 表）
        """
        self.db = db

    # ──────────── 用户管理 ────────────

    def register(self, username: str, password: str, display_name: str = None) -> dict:
        """
        注册新用户

        Returns:
            {"id": ..., "username": ..., "display_name": ...}

        Raises:
            ValueError: 用户名已存在或参数无效
        """
        username = username.strip().lower()
        if not username or len(username) < 2:
            raise ValueError("用户名至少 2 个字符")
        if not password or len(password) < 4:
            raise ValueError("密码至少 4 个字符")

        # 检查重复
        cursor = self.db.conn.cursor()
        cursor.execute("SELECT id FROM auth_users WHERE username = ?", (username,))
        if cursor.fetchone():
            raise ValueError(f"用户名 '{username}' 已被注册")

        user_id = str(uuid.uuid4())
        password_hash = _hash_password(password)
        display_name = display_name or username

        cursor.execute(
            "INSERT INTO auth_users (id, username, password_hash, display_name) VALUES (?, ?, ?, ?)",
            (user_id, username, password_hash, display_name),
        )
        self.db.conn.commit()
        logger.info(f"新用户注册: id={user_id[:8]}, username={username}")

        return {"id": user_id, "username": username, "display_name": display_name}

    def authenticate(self, username: str, password: str) -> Optional[dict]:
        """
        验证用户名密码

        Returns:
            用户 dict 或 None（验证失败）
        """
        username = username.strip().lower()
        cursor = self.db.conn.cursor()
        cursor.execute(
            "SELECT id, username, password_hash, display_name, created_at FROM auth_users WHERE username = ?",
            (username,),
        )
        row = cursor.fetchone()
        if not row:
            return None

        if not _verify_password(password, row["password_hash"]):
            return None

        return {
            "id": row["id"],
            "username": row["username"],
            "display_name": row["display_name"],
            "created_at": str(row["created_at"]),
        }

    def get_user_by_id(self, user_id: str) -> Optional[dict]:
        """根据 ID 获取用户"""
        cursor = self.db.conn.cursor()
        cursor.execute(
            "SELECT id, username, display_name, created_at FROM auth_users WHERE id = ?",
            (user_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    # ──────────── JWT ────────────

    def create_token(self, user_id: str, username: str) -> str:
        """签发 JWT token"""
        expire = datetime.now(timezone.utc) + timedelta(days=JWT_EXPIRE_DAYS)
        payload = {
            "sub": user_id,
            "username": username,
            "exp": expire,
        }
        return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

    def verify_token(self, token: str) -> Optional[dict]:
        """
        校验 JWT token

        Returns:
            {"user_id": ..., "username": ...} 或 None
        """
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            user_id = payload.get("sub")
            username = payload.get("username")
            if not user_id:
                return None
            return {"user_id": user_id, "username": username}
        except JWTError:
            return None


# ──────────── FastAPI 依赖 ────────────

# 全局 AuthService 实例（在 server.py 中设置）
_auth_service: Optional[AuthService] = None


def set_auth_service(service: AuthService):
    """设置全局 AuthService 实例"""
    global _auth_service
    _auth_service = service


def get_auth_service() -> AuthService:
    """获取全局 AuthService 实例"""
    if _auth_service is None:
        raise RuntimeError("AuthService 未初始化")
    return _auth_service


async def get_current_user(authorization: str = None) -> dict:
    """
    FastAPI 依赖：从 Authorization header 解析当前用户

    用法：
        @app.get("/api/xxx")
        async def handler(user: dict = Depends(get_current_user)):
            user_id = user["user_id"]

    Returns:
        {"user_id": ..., "username": ...}

    Raises:
        HTTPException 401 如果未认证
    """
    from fastapi import HTTPException, Header

    # 从 header 提取 token
    if not authorization:
        raise HTTPException(status_code=401, detail="未提供认证信息")

    # 支持 "Bearer <token>" 格式
    token = authorization
    if token.startswith("Bearer "):
        token = token[7:]

    auth = get_auth_service()
    result = auth.verify_token(token)
    if not result:
        raise HTTPException(status_code=401, detail="认证已过期或无效，请重新登录")

    return result