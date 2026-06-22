"""
dogAgent FastAPI 服务

启动：
  cd dogAgent
  python -m uvicorn api.server:app --reload --port 8000

访问：
  Web UI:  http://localhost:8000
  API:     http://localhost:8000/api/docs
"""

import os
import sys
import logging

# 确保项目根目录在 sys.path
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from dotenv import load_dotenv

load_dotenv(os.path.join(PROJECT_DIR, ".env"))

from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from typing import Optional
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from api.models import (
    ChatRequest,
    ChatResponse,
    SessionListResponse,
    SessionInfo,
    CreateSessionResponse,
    MessagesResponse,
    MessageInfo,
    PreferencesResponse,
    PreferenceInfo,
    AddPreferenceRequest,
    HealthResponse,
    RegisterRequest,
    LoginRequest,
    TokenResponse,
    UserInfo,
)
from api.auth import AuthService, set_auth_service, get_current_user
from api.a2a import a2a_app, setup_a2a

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("api")

# === 全局单例 ===
_llm = None
_retriever = None
_memory = None
_auth = None
_bg_tasks = None
_judge = None
_staging = None
_a2a_tool = None


def _init_components():
    """初始化 LLM / Retriever / Memory / Auth / BackgroundTasks / Staging（懒加载单例）"""
    global _llm, _retriever, _memory, _auth, _bg_tasks, _judge, _staging, _a2a_tool

    from agent.chat import create_retriever, create_memory_system
    from agent.llm import create_main_llm

    if _llm is None:
        _llm = create_main_llm()
        logger.info("LLM 已初始化")

    if _retriever is None:
        _retriever = create_retriever(llm=_llm)
        logger.info(f"Retriever 已初始化，索引条目数: {len(_retriever.index) if _retriever.index else 0}")

    if _memory is None:
        try:
            _memory = create_memory_system()
            from agent.compaction import CompactionEngine
            from agent.query_rewrite import QueryRewriter

            _memory["compaction"] = CompactionEngine(
                _memory["conv_store"], _memory["summary_dag"], _llm
            )
            _memory["query_rewriter"] = QueryRewriter(_llm)
            logger.info("记忆系统已初始化")
        except Exception as e:
            logger.warning(f"记忆系统初始化失败: {e}")
            _memory = None

    # 初始化后台任务管理器
    if _bg_tasks is None:
        from agent.background import BackgroundTaskManager
        _bg_tasks = BackgroundTaskManager()
        logger.info("后台任务管理器已初始化")

    # 初始化 judge（使用 qwen-max，与主 LLM 共用实例）
    if _judge is None:
        from agent.judge import ResponseJudge
        _judge = ResponseJudge(_llm)
        logger.info("ResponseJudge 已初始化（qwen-max）")

    # 初始化 staging 暂存层 + 后台超时 promote 线程
    if _staging is None and _memory is not None:
        from agent.staging import StagingStore
        import threading
        _staging = StagingStore(_memory["db"], _memory["conv_store"])
        t = threading.Thread(target=_staging_drain_loop, daemon=True)
        t.start()
        logger.info("StagingStore 已初始化，超时 promote 线程已启动")

    # 初始化认证服务
    if _auth is None and _memory is not None:
        _auth = AuthService(_memory["db"])
        set_auth_service(_auth)
        logger.info("认证服务已初始化")

    # 注入 A2A 共享状态
    if _llm is not None and _retriever is not None:
        setup_a2a(_llm, _retriever)
        logger.info("A2A endpoint 已配置")

    # 初始化 A2A 客户端工具（若配置了外部 agent）
    if _a2a_tool is None:
        a2a_agent_url = os.getenv("A2A_AGENT_URL", "").strip()
        if a2a_agent_url:
            from agent.a2a_client import create_a2a_tool
            _a2a_tool = create_a2a_tool(a2a_agent_url)
            logger.info(f"A2A 客户端工具已注册，目标: {a2a_agent_url}（懒加载，首次调用时连接）")


def _staging_drain_loop():
    """后台线程：每5分钟扫描 staging，将超过 30 分钟的轮次强制 promote 到持久层。"""
    import time
    while True:
        time.sleep(300)
        if not _staging or not _memory:
            continue
        try:
            expired = _staging.get_expired()
            for turn in expired:
                _staging.promote_turn(
                    turn["user_id"], turn["session_id"],
                    turn["turn_index"], _memory["conv_store"],
                )
                if _bg_tasks:
                    _bg_tasks.submit_preference_extract(
                        _llm, _memory["user_prefs"],
                        turn["user_id"], turn["session_id"],
                        turn["user_content"], turn["assistant_content"],
                    )
                    _bg_tasks.submit_compaction(
                        _memory.get("compaction"),
                        turn["user_id"], turn["session_id"],
                    )
        except Exception as e:
            logger.warning(f"Staging drain loop error: {e}")


def _evaluate_staging(user_id: str, session_id: str, current_user_input: str):
    """对 staging 中所有轮次做 batch judge，SKIP 的立即删除，STORE 的继续留在 staging。"""
    if not _staging or not _judge:
        return
    turns = _staging.get_turns(user_id, session_id)
    if not turns:
        return
    verdicts = _judge.batch_judge(turns, follow_up=current_user_input)
    for turn, store in zip(turns, verdicts):
        if not store:
            _staging.delete_turn(user_id, session_id, turn["turn_index"])
            logger.info(f"Staging SKIP: session={session_id} turn={turn['turn_index']} q={turn['user_content'][:30]!r}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动时初始化组件"""
    _init_components()
    yield


app = FastAPI(
    title="dogAgent API",
    description="雪纳瑞犬知识助手 API",
    version="1.0.0",
    lifespan=lifespan,
)

# 静态文件（Web UI）
WEB_DIR = os.path.join(PROJECT_DIR, "web")
if os.path.isdir(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

# A2A 子应用
app.mount("/a2a", a2a_app)


# ──────────────────────────────────────────────
# 辅助：从 Header 获取当前用户
# ──────────────────────────────────────────────


async def get_user_from_header(authorization: Optional[str] = Header(default=None)) -> dict:
    """从 Authorization header 解析当前用户，未认证则 401"""
    return await get_current_user(authorization)


async def get_optional_user(authorization: Optional[str] = Header(default=None)) -> Optional[dict]:
    """尝试从 Authorization header 解析用户，未认证返回 None（不报错）"""
    if not authorization:
        return None
    try:
        return await get_current_user(authorization)
    except HTTPException:
        return None


# ──────────────────────────────────────────────
# Web UI
# ──────────────────────────────────────────────


@app.get("/", include_in_schema=False)
async def serve_ui():
    """Serve Web UI"""
    index_path = os.path.join(WEB_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "dogAgent API is running. Web UI not found at /web/index.html"}


# ──────────────────────────────────────────────
# 认证端点
# ──────────────────────────────────────────────


@app.post("/api/auth/register", response_model=TokenResponse)
async def register(req: RegisterRequest):
    """注册新用户"""
    if not _auth:
        raise HTTPException(status_code=503, detail="认证服务不可用")
    try:
        user = _auth.register(req.username, req.password, req.display_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    token = _auth.create_token(user["id"], user["username"])
    return TokenResponse(
        token=token,
        user_id=user["id"],
        username=user["username"],
        display_name=user["display_name"],
    )


@app.post("/api/auth/login", response_model=TokenResponse)
async def login(req: LoginRequest):
    """用户登录"""
    if not _auth:
        raise HTTPException(status_code=503, detail="认证服务不可用")

    user = _auth.authenticate(req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    token = _auth.create_token(user["id"], user["username"])
    return TokenResponse(
        token=token,
        user_id=user["id"],
        username=user["username"],
        display_name=user["display_name"],
    )


@app.get("/api/auth/me", response_model=UserInfo)
async def get_me(user: dict = Depends(get_user_from_header)):
    """获取当前用户信息"""
    full_user = _auth.get_user_by_id(user["user_id"]) if _auth else None
    return UserInfo(
        user_id=user["user_id"],
        username=user["username"],
        display_name=full_user["display_name"] if full_user else user["username"],
        created_at=str(full_user["created_at"]) if full_user else None,
    )


# ──────────────────────────────────────────────
# API 端点
# ──────────────────────────────────────────────


@app.get("/api/health", response_model=HealthResponse)
async def health():
    """健康检查"""
    wiki_count = len(_retriever.index) if _retriever and _retriever.index else 0
    return HealthResponse(
        status="ok",
        wiki_articles=wiki_count,
        memory_available=_memory is not None,
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, user: dict = Depends(get_user_from_header)):
    """核心对话接口（需认证）"""
    import time
    from agent.chat import (
        SYSTEM_PROMPT,
        build_memory_context,
        build_messages,
        create_memory_expand_tool,
        create_web_search_tool,
        create_image_search_tool,
        invoke_with_tools,
    )
    from langchain_core.messages import HumanMessage

    if not _llm or not _retriever:
        raise HTTPException(status_code=503, detail="服务未就绪，LLM 或 Retriever 未初始化")

    t0 = time.perf_counter()

    user_id = user["user_id"]
    session_id = req.session_id
    user_input = req.message.strip()
    image_base64 = req.image_base64

    if not user_input:
        raise HTTPException(status_code=400, detail="消息不能为空")

    # 确保 session
    if _memory:
        if not session_id:
            session_id = _memory["session_mgr"].ensure_session(user_id)
        else:
            s = _memory["session_mgr"].get_session(session_id)
            if not s or s.get("user_id") != user_id:
                session_id = _memory["session_mgr"].ensure_session(user_id)
    else:
        session_id = session_id or "no-memory"

    # 1. Query rewrite
    rewritten_query = user_input
    recent_msgs = []
    if _memory and session_id != "no-memory":
        recent_msgs = _memory["conv_store"].get_recent(user_id, session_id, limit=10)
        if _memory.get("query_rewriter"):
            rewritten_query = _memory["query_rewriter"].rewrite(user_input, recent_msgs)
    t1 = time.perf_counter()
    logger.info(f"[timing] query_rewrite={t1-t0:.3f}s")

    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    async def _run_parallel_preprocess():
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=2) as pool:
            staging_fut = loop.run_in_executor(
                pool, _evaluate_staging, user_id, session_id, user_input
            ) if (_staging and _memory and session_id != "no-memory") else asyncio.sleep(0)
            wiki_fut = loop.run_in_executor(
                pool, _retriever.retrieve, rewritten_query, 3
            )
            _, wiki_results = await asyncio.gather(staging_fut, wiki_fut)
        return wiki_results

    results = await _run_parallel_preprocess()
    wiki_context = _retriever.format_context(results)
    sources = [r.title for r in results]
    t2 = time.perf_counter()
    logger.info(f"[timing] parallel(staging_judge + wiki_retrieval)={t2-t1:.3f}s")

    # 2.5 购物意图检测 → 强制执行 PDD 搜索，结果注入 context
    PDD_INTENT_KEYWORDS = ["在哪买", "哪里买", "怎么买", "购买", "推荐买", "多少钱", "价格",
                           "拼多多", "pdd", "搜", "找", "哪款", "选购", "入手",
                           "推荐狗粮", "推荐食品", "推荐零食", "品牌推荐", "哪个品牌"]
    query_lower = user_input.lower()
    pdd_context = ""
    if any(kw in query_lower for kw in PDD_INTENT_KEYWORDS):
        try:
            from agent.pdd_tool import create_pdd_search_tool
            import re
            keyword = re.sub(r"(帮我|我想|请|在拼多多|在pdd上?|搜索?|找|推荐|买|购买|多少钱|价格|哪里|哪款|\d+个)", "", user_input).strip()
            if not keyword:
                keyword = rewritten_query
            pdd_tool_fn = create_pdd_search_tool()
            pdd_result = pdd_tool_fn.invoke({"keyword": keyword, "sort_type": 0})
            if pdd_result and ("拼多多搜索结果" in pdd_result or "拼多多推荐结果" in pdd_result):
                pdd_context = f"\n\n【拼多多实时搜索结果 — 以下数据来自 PDD API，必须原文展示给用户，不得改写或替换任何链接】\n{pdd_result}"
                logger.info(f"PDD 强制搜索完成，关键词: {keyword}")
        except Exception as e:
            logger.warning(f"PDD 强制搜索失败: {e}")
    t3 = time.perf_counter()
    logger.info(f"[timing] pdd_forced_search={t3-t2:.3f}s  (skipped={not pdd_context})")

    # 3. 记忆上下文
    memory_context = ""
    if _memory and session_id != "no-memory":
        memory_context = build_memory_context(_memory, user_id, session_id)

    # 4. 组装消息：持久层 tail + staging 暂存消息（staging 排最后，最近发生）
    if _memory and session_id != "no-memory":
        tail_msgs = _memory["summary_dag"].get_context_messages(user_id, session_id)
        if _staging:
            tail_msgs = tail_msgs + _staging.get_messages(user_id, session_id)
    else:
        tail_msgs = recent_msgs[-6:] if recent_msgs else []
    messages = build_messages(SYSTEM_PROMPT, memory_context, wiki_context + pdd_context, tail_msgs, user_input)
    t4 = time.perf_counter()
    logger.info(f"[timing] memory_ctx + msg_assembly={t4-t3:.3f}s")

    # 如果用户上传了图片，替换最后一条 HumanMessage 为多模态格式
    if image_base64:
        messages = messages[:-1]
        messages.append(
            HumanMessage(
                content=[
                    {"type": "image_url", "image_url": {"url": image_base64}},
                    {"type": "text", "text": user_input},
                ]
            )
        )

    try:
        # 有图片时切换到视觉模型
        active_llm = _llm
        if image_base64:
            try:
                from agent.llm import create_vision_llm
                active_llm = create_vision_llm()
            except Exception as e:
                logger.warning(f"视觉模型初始化失败，降级到普通 LLM: {e}")

        search_tool = create_web_search_tool()
        img_tool = create_image_search_tool()
        from agent.pdd_tool import create_pdd_search_tool
        pdd_tool = create_pdd_search_tool()
        extra_tools = [_a2a_tool] if _a2a_tool else []
        if _memory and session_id != "no-memory":
            expand_tool = create_memory_expand_tool(_memory, user_id)
            answer, images = invoke_with_tools(active_llm, messages, [expand_tool, search_tool, img_tool, pdd_tool] + extra_tools)
        else:
            answer, images = invoke_with_tools(active_llm, messages, [search_tool, img_tool, pdd_tool] + extra_tools)
    except Exception as e:
        logger.error(f"LLM 调用失败: {e}")
        raise HTTPException(status_code=500, detail=f"LLM 调用失败: {e}")
    t5 = time.perf_counter()
    logger.info(f"[timing] main_llm(react_agent)={t5-t4:.3f}s")

    # 5. 写入 staging
    if _staging and _memory and session_id != "no-memory":
        _staging.add_turn(user_id, session_id, user_input, answer)
        _memory["session_mgr"].touch_session(session_id)
        if _bg_tasks:
            _bg_tasks.submit_title_generate(
                _llm, _memory["session_mgr"], session_id
            )

    logger.info(f"[timing] total={time.perf_counter()-t0:.3f}s  session={session_id[:8]}")

    return ChatResponse(
        answer=answer,
        session_id=session_id,
        sources=sources,
        images=images,
        rewritten_query=rewritten_query if rewritten_query != user_input else None,
    )


# ──────────────────────────────────────────────
# 会话管理（需认证）
# ──────────────────────────────────────────────


@app.get("/api/sessions", response_model=SessionListResponse)
async def list_sessions(user: dict = Depends(get_user_from_header)):
    """列出当前用户所有会话"""
    if not _memory:
        return SessionListResponse(sessions=[])
    user_id = user["user_id"]
    raw = _memory["session_mgr"].list_sessions(user_id)
    sessions = [
        SessionInfo(
            id=s["id"],
            title=s.get("title"),
            created_at=str(s.get("created_at", "")),
            updated_at=str(s.get("updated_at", "")),
            is_active=s.get("is_active", True),
        )
        for s in raw
    ]
    return SessionListResponse(sessions=sessions)


@app.post("/api/sessions", response_model=CreateSessionResponse)
async def create_session(user: dict = Depends(get_user_from_header)):
    """创建新会话"""
    if not _memory:
        raise HTTPException(status_code=503, detail="记忆系统不可用")
    sid = _memory["session_mgr"].create_session(user["user_id"])
    return CreateSessionResponse(session_id=sid)


@app.get("/api/sessions/{session_id}/messages", response_model=MessagesResponse)
async def get_messages(
    session_id: str,
    limit: int = 50,
    user: dict = Depends(get_user_from_header),
):
    """获取会话历史消息（只能查看自己的会话）"""
    if not _memory:
        return MessagesResponse(session_id=session_id, messages=[])

    user_id = user["user_id"]

    # 权限校验：确保 session 属于当前用户
    s = _memory["session_mgr"].get_session(session_id)
    if not s or s.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="无权访问该会话")

    # 查看历史前先 drain staging，确保用户看到完整历史
    if _staging:
        try:
            _staging.drain_session(
                user_id, session_id, _memory["conv_store"],
                bg_tasks=_bg_tasks, llm=_llm,
                user_prefs=_memory["user_prefs"],
                compaction=_memory.get("compaction"),
            )
        except Exception as e:
            logger.warning(f"Staging drain on messages view failed: {e}")

    # 从 context_items 取当前视图（摘要 + 消息，按 ordinal 顺序），截取最近 limit 条
    items = _memory["summary_dag"].get_context_items(user_id, session_id)
    msgs = []
    for item in items:
        if item["item_type"] == "message":
            cursor = _memory["db"].conn.cursor()
            cursor.execute(
                "SELECT role, content, created_at FROM conversations WHERE id = ?",
                (item["message_id"],),
            )
            row = cursor.fetchone()
            if row:
                msgs.append(MessageInfo(
                    role=row["role"],
                    content=row["content"],
                    created_at=str(row["created_at"] or ""),
                ))
        elif item["item_type"] == "summary":
            cursor = _memory["db"].conn.cursor()
            cursor.execute(
                "SELECT content, created_at FROM summaries WHERE id = ?",
                (item["summary_id"],),
            )
            row = cursor.fetchone()
            if row:
                msgs.append(MessageInfo(
                    role="summary",
                    content=row["content"],
                    created_at=str(row["created_at"] or ""),
                ))
    return MessagesResponse(session_id=session_id, messages=msgs[-limit:])


@app.delete("/api/sessions/{session_id}")
async def delete_session(
    session_id: str,
    user: dict = Depends(get_user_from_header),
):
    """删除（归档）会话"""
    if not _memory:
        raise HTTPException(status_code=503, detail="记忆系统不可用")

    user_id = user["user_id"]
    s = _memory["session_mgr"].get_session(session_id)
    if not s or s.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="无权操作该会话")

    _memory["session_mgr"].archive_session(session_id)
    return {"status": "ok", "session_id": session_id}


# ──────────────────────────────────────────────
# 偏好管理（需认证）
# ──────────────────────────────────────────────


@app.get("/api/preferences", response_model=PreferencesResponse)
async def get_preferences(user: dict = Depends(get_user_from_header)):
    """获取当前用户偏好"""
    if not _memory:
        return PreferencesResponse(preferences=[])
    raw = _memory["user_prefs"].get_active(user["user_id"])
    prefs = [
        PreferenceInfo(
            id=p.get("id", 0),
            content=p["content"],
            created_at=str(p.get("created_at", "")),
        )
        for p in raw
    ]
    return PreferencesResponse(preferences=prefs)


@app.post("/api/preferences")
async def add_preference(req: AddPreferenceRequest, user: dict = Depends(get_user_from_header)):
    """添加用户偏好"""
    if not _memory:
        raise HTTPException(status_code=503, detail="记忆系统不可用")
    _memory["user_prefs"].add_preference(user["user_id"], req.content)
    return {"status": "ok", "content": req.content}


# ──────────────────────────────────────────────
# 启动入口
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)