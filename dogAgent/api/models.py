"""Pydantic 请求/响应模型"""

from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    message: str = Field(..., description="用户消息")
    user_id: str = Field(default="default_user", description="用户ID")
    session_id: Optional[str] = Field(default=None, description="会话ID，空则自动选最近的")


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    sources: list[str] = Field(default_factory=list)
    rewritten_query: Optional[str] = None


class SessionInfo(BaseModel):
    id: str
    title: Optional[str] = None
    created_at: str
    updated_at: str
    is_active: bool = True


class SessionListResponse(BaseModel):
    sessions: list[SessionInfo]


class CreateSessionResponse(BaseModel):
    session_id: str


class MessageInfo(BaseModel):
    role: str
    content: str
    created_at: str


class MessagesResponse(BaseModel):
    session_id: str
    messages: list[MessageInfo]


class PreferenceInfo(BaseModel):
    id: int
    content: str
    created_at: str


class PreferencesResponse(BaseModel):
    preferences: list[PreferenceInfo]


class AddPreferenceRequest(BaseModel):
    content: str = Field(..., description="偏好内容")
    user_id: str = Field(default="default_user")


class HealthResponse(BaseModel):
    status: str = "ok"
    wiki_articles: int = 0
    memory_available: bool = False


# ──────────── 认证相关 ────────────


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=2, max_length=50, description="用户名")
    password: str = Field(..., min_length=4, max_length=100, description="密码")
    display_name: Optional[str] = Field(default=None, description="显示名称")


class LoginRequest(BaseModel):
    username: str = Field(..., description="用户名")
    password: str = Field(..., description="密码")


class TokenResponse(BaseModel):
    token: str
    user_id: str
    username: str
    display_name: Optional[str] = None


class UserInfo(BaseModel):
    user_id: str
    username: str
    display_name: Optional[str] = None
    created_at: Optional[str] = None
