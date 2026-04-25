"""
dogAgent 对话模块

基于 Wiki 知识库的 RAG 对话 Agent，集成多层记忆系统。

功能：
- Wiki 知识库检索（RAG）
- 多轮对话记忆（ConversationStore + DAG 压缩）
- 多会话管理（SessionManager）
- 用户偏好记忆（UserPreferences）
- 跨 session 记忆检索（MemoryIndex）
- 查询重写（QueryRewriter）

用法:
  python agent/chat.py              # 交互模式（带记忆）
  python agent/chat.py --no-memory  # 无记忆模式（兼容旧版）
  python agent/chat.py "问题"        # 单次查询
"""

import os
import sys
import logging

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("chat")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIKI_DIR = os.path.join(PROJECT_DIR, "wiki")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "qwen-plus")
MEMORY_DB_PATH = os.path.join(PROJECT_DIR, "data", "memory.db")
DEFAULT_USER_ID = os.environ.get("DOG_AGENT_USER", "default_user")

SYSTEM_PROMPT = """你是 dogAgent — 一个专业的雪纳瑞犬知识助手，同时能记住用户的个人信息。

你有三类信息来源，使用规则不同：

【记忆（用户偏好 / 对话摘要）】
- 来自系统注入的"用户偏好"和"之前的对话摘要"部分
- 这是关于当前用户的真实可靠信息，直接使用，无需引用来源
- 用于回答关于用户本人的问题（姓名、宠物、历史等）

【知识库】
- 来自根据问题检索到的 Wiki 文档片段
- 用于回答关于雪纳瑞犬的专业问题

【网页搜索（web_search 工具）— 重要！】
- 当用户要求推荐视频、教程、产品、链接，或者知识库内容不足时，你**必须**调用 web_search 工具搜索互联网
- 不要拒绝提供链接，不要说"我无法访问网络" — 你有 web_search 工具，直接用它搜索
- 搜索后把找到的链接和信息直接给用户，附上标题和 URL

回答时请：
1. 优先使用记忆中的用户信息回答个人相关问题
2. 使用知识库内容回答雪纳瑞专业问题
3. 用户要求链接/推荐/最新信息时，**立即调用 web_search**，不要犹豫
4. 用友好、专业的语气回答
5. 涉及健康问题时，建议咨询兽医
"""

CONTEXT_TEMPLATE = """以下是根据用户问题检索到的相关知识库内容：

{context}

---
请基于以上内容回答用户的问题。如果内容不足以回答，请说明。
"""


def create_llm():
    """创建 LLM 实例（委托给 agent.llm）"""
    try:
        from agent.llm import create_llm as _create
    except ImportError:
        from llm import create_llm as _create
    return _create()


def create_retriever(llm=None):
    """
    创建检索器实例。

    如果提供了 llm，使用 LLMWikiIndexRetriever（Karpathy 风格，LLM 读 index 判断相关性）。
    否则使用 WikiRetriever（规则匹配 fallback）。
    """
    try:
        from agent.retriever import LLMWikiIndexRetriever, WikiRetriever
    except ImportError:
        from retriever import LLMWikiIndexRetriever, WikiRetriever

    if llm:
        retriever = LLMWikiIndexRetriever(llm=llm)
        logger.info("使用 LLMWikiIndexRetriever（Karpathy 风格）")
    else:
        retriever = WikiRetriever()

    if not retriever.index:
        logger.warning("索引为空，尝试重新生成...")
        try:
            try:
                from agent.build_index import main as build_index_main
            except ImportError:
                from build_index import main as build_index_main

            build_index_main()
            # 重新创建检索器（加载新索引）
            if llm:
                retriever = LLMWikiIndexRetriever(llm=llm)
            else:
                retriever = WikiRetriever()
        except Exception as e:
            logger.error(f"索引生成失败: {e}")

    return retriever


def create_memory_system():
    """创建记忆系统（所有组件）"""
    try:
        from agent.memory import MemoryDB, ConversationStore, UserPreferences, SummaryDAG
        from agent.session import SessionManager
        from agent.compaction import CompactionEngine
        from agent.query_rewrite import QueryRewriter
    except ImportError:
        from memory import MemoryDB, ConversationStore, UserPreferences, SummaryDAG
        from session import SessionManager
        from compaction import CompactionEngine
        from query_rewrite import QueryRewriter

    # 确保 data 目录存在
    os.makedirs(os.path.dirname(MEMORY_DB_PATH), exist_ok=True)

    db = MemoryDB(MEMORY_DB_PATH)
    conv_store = ConversationStore(db)
    user_prefs = UserPreferences(db)
    summary_dag = SummaryDAG(db)
    session_mgr = SessionManager(db)

    # DAG 钻取引擎（Lossless Claw 回溯）
    try:
        from agent.memory_drilldown import MemoryDrillDown
    except ImportError:
        from memory_drilldown import MemoryDrillDown
    drilldown = MemoryDrillDown()

    return {
        "db": db,
        "conv_store": conv_store,
        "user_prefs": user_prefs,
        "summary_dag": summary_dag,
        "session_mgr": session_mgr,
        "drilldown": drilldown,
        "query_rewriter": None,  # 在 chat_loop 中设置（需要 llm）
        "compaction": None,  # 在 chat_loop 中设置（需要 llm）
    }


def build_memory_context(memory: dict, user_id: str, session_id: str) -> str:
    """
    组装记忆上下文（插入到 system prompt 之后、wiki 上下文之前）。

    包含：
    1. DAG 摘要（XML 格式，LLM 自行通过 memory_expand tool 决定是否展开）
    2. 用户偏好
    """
    parts = []

    # 1. 当前 session 的 DAG 摘要（XML 格式）
    xml_ctx = memory["summary_dag"].get_context_text(user_id, session_id)
    if xml_ctx:
        parts.append("## 之前的对话摘要（如需细节可使用 memory_expand 工具展开）")
        parts.append(xml_ctx)

    # 2. 用户偏好
    pref_text = memory["user_prefs"].get_active_text(user_id)
    if pref_text:
        parts.append("## 用户偏好")
        parts.append(pref_text)
        memory["user_prefs"].touch_preferences(user_id)

    return "\n\n".join(parts)


def create_memory_expand_tool(memory: dict, user_id: str):
    """创建 memory_expand LangChain tool，LLM 调用后触发 DAG 钻取。"""
    from langchain_core.tools import tool

    drilldown = memory.get("drilldown")
    summary_dag = memory.get("summary_dag")
    conv_store = memory.get("conv_store")

    @tool
    def memory_expand(summary_id: str) -> str:
        """展开历史对话摘要，获取更详细的原始内容。
        当摘要提到了某个话题但细节不足时调用（如具体药名、剂量、日期等）。
        参数 summary_id 来自 <summary id="..."> 标签中的 id 属性。
        """
        if not drilldown or not summary_dag or not conv_store:
            return "（记忆系统不可用）"
        result = drilldown.drilldown_by_id(summary_id, user_id, summary_dag, conv_store)
        return result or "（未找到相关详情）"

    return memory_expand


def create_web_search_tool():
    """创建 web_search LangChain tool，LLM 判断知识库不足时调用。"""
    from langchain_core.tools import tool
    import requests

    @tool
    def web_search(query: str) -> str:
        """搜索互联网获取最新信息。
        当知识库中没有足够信息回答用户问题时调用，比如：
        - 用户要求推荐具体的产品、视频、教程链接
        - 需要最新的价格、政策等实时信息
        - 知识库未覆盖的话题
        参数 query: 搜索关键词（中文或英文）
        """
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            # 使用 DuckDuckGo HTML 搜索（无需 API key）
            resp = requests.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
                headers=headers,
                timeout=10,
            )
            if resp.status_code != 200:
                return f"搜索失败（HTTP {resp.status_code}）"

            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, "html.parser")
            results = []
            for r in soup.select(".result")[:5]:
                title_el = r.select_one(".result__a")
                snippet_el = r.select_one(".result__snippet")
                if title_el:
                    title = title_el.get_text(strip=True)
                    url = title_el.get("href", "")
                    snippet = snippet_el.get_text(strip=True) if snippet_el else ""
                    results.append(f"- {title}\n  {snippet}\n  {url}")

            if not results:
                return "未找到相关搜索结果"
            return "搜索结果：\n\n" + "\n\n".join(results)

        except Exception as e:
            return f"搜索出错: {e}"

    return web_search


def invoke_with_tools(llm, messages: list, tools: list, max_tool_calls: int = 3) -> str:
    """带 tool calling 的 LLM 调用循环。"""
    from langchain_core.messages import ToolMessage

    llm_with_tools = llm.bind_tools(tools)
    tool_map = {t.name: t for t in tools}
    msgs = list(messages)

    for _ in range(max_tool_calls):
        response = llm_with_tools.invoke(msgs)
        if not response.tool_calls:
            return response.content
        msgs.append(response)
        for tc in response.tool_calls:
            tool_fn = tool_map.get(tc["name"])
            result = tool_fn.invoke(tc["args"]) if tool_fn else f"（未知工具: {tc['name']}）"
            msgs.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    # 超过最大轮数，强制最终回答（不带 tools）
    return llm.invoke(msgs).content


def build_messages(
    system_prompt: str,
    memory_context: str,
    wiki_context: str,
    recent_messages: list[dict],
    user_input: str,
) -> list:
    """
    组装最终的 LLM 消息列表。

    结构：
    [SystemMessage, (memory context), recent history..., (wiki context + user question)]
    """
    msgs = []

    # System prompt（含记忆上下文）
    full_system = system_prompt
    if memory_context:
        full_system += "\n\n" + memory_context
    msgs.append(SystemMessage(content=full_system))

    # 最近对话历史（作为多轮 context）
    for m in recent_messages:
        if m["role"] == "user":
            msgs.append(HumanMessage(content=m["content"]))
        else:
            msgs.append(AIMessage(content=m["content"]))

    # Wiki 检索上下文 + 用户问题
    if wiki_context:
        user_content = CONTEXT_TEMPLATE.format(context=wiki_context) + f"\n\n用户问题: {user_input}"
    else:
        user_content = user_input
    msgs.append(HumanMessage(content=user_content))

    return msgs


def chat_loop():
    """交互式对话循环（带记忆系统）"""
    use_memory = "--no-memory" not in sys.argv

    print("\n🐾 dogAgent — 雪纳瑞知识助手")
    if use_memory:
        print("   📝 记忆模式已启用")
    print("=" * 50)
    print("命令: quit/exit 退出 | debug 调试 | /new 新会话 | /sessions 列表")
    print()

    # 创建核心组件
    llm = create_llm()
    retriever = create_retriever(llm=llm)

    # 记忆系统
    memory = None
    session_id = None
    user_id = DEFAULT_USER_ID

    if use_memory:
        try:
            memory = create_memory_system()
            # 注入 LLM 到需要的组件
            try:
                from agent.compaction import CompactionEngine
                from agent.query_rewrite import QueryRewriter
            except ImportError:
                from compaction import CompactionEngine
                from query_rewrite import QueryRewriter

            memory["compaction"] = CompactionEngine(
                memory["conv_store"], memory["summary_dag"], llm
            )
            memory["query_rewriter"] = QueryRewriter(llm)

            # 获取或创建 session
            session_id = memory["session_mgr"].ensure_session(user_id)
            session = memory["session_mgr"].get_session(session_id)
            title = session.get("title", "新会话") if session else "新会话"
            print(f"  📂 会话: {title} ({session_id[:8]}...)")
            logger.info(f"会话就绪: {session_id}")
        except Exception as e:
            logger.error(f"记忆系统初始化失败: {e}")
            memory = None
            print("  ⚠️  记忆系统不可用，使用无状态模式")

    # 后台任务管理器（偏好提取/压缩/标题生成，不阻塞用户）
    bg_tasks = None
    try:
        from agent.background import BackgroundTaskManager
    except ImportError:
        from background import BackgroundTaskManager
    bg_tasks = BackgroundTaskManager()

    debug_mode = False

    while True:
        try:
            user_input = input("🧑 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n再见！🐶")
            if bg_tasks:
                bg_tasks.shutdown()
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q", "退出"):
            if bg_tasks:
                bg_tasks.shutdown()
            print("\n再见！🐶")
            break
        if user_input.lower() == "debug":
            debug_mode = not debug_mode
            print(f"  调试模式: {'开启' if debug_mode else '关闭'}")
            continue

        # === 会话管理命令 ===
        if user_input.startswith("/"):
            if memory:
                session_id = _handle_command(user_input, memory, user_id, session_id)
            else:
                print("  ⚠️  记忆系统未启用")
            continue

        # === 主对话流程 ===

        # 1. 查询重写
        rewritten_query = user_input
        recent_msgs = []
        if memory and session_id:
            recent_msgs = memory["conv_store"].get_recent(user_id, session_id, limit=10)
            if memory["query_rewriter"]:
                rewritten_query = memory["query_rewriter"].rewrite(user_input, recent_msgs)
                if rewritten_query != user_input and debug_mode:
                    print(f"  🔄 重写: {rewritten_query}")

        # 2. Wiki 知识检索（用重写后的查询）
        results = retriever.retrieve(rewritten_query, top_k=3)

        if debug_mode:
            print(f"\n  📚 检索到 {len(results)} 篇相关文档:")
            for i, r in enumerate(results, 1):
                print(f"    #{i} [{r.score:.3f}] {r.title}")
                for reason in r.match_reasons:
                    print(f"        ✓ {reason}")
            print()

        wiki_context = retriever.format_context(results)

        # 3. 组装记忆上下文
        memory_context = ""
        if memory and session_id:
            memory_context = build_memory_context(memory, user_id, session_id)
            if debug_mode and memory_context:
                print(f"  🧠 记忆上下文: {len(memory_context)} 字符")

        # 4. 组装消息并调用 LLM
        # fresh tail 消息直接从 context_items 取，与摘要保持同一视图
        if memory and session_id:
            tail_msgs = memory["summary_dag"].get_context_messages(user_id, session_id)
        else:
            tail_msgs = recent_msgs[-6:] if recent_msgs else []
        messages = build_messages(SYSTEM_PROMPT, memory_context, wiki_context, tail_msgs, user_input)

        try:
            if memory and session_id:
                expand_tool = create_memory_expand_tool(memory, user_id)
                search_tool = create_web_search_tool()
                answer = invoke_with_tools(llm, messages, [expand_tool, search_tool])
            else:
                search_tool = create_web_search_tool()
                answer = invoke_with_tools(llm, messages, [search_tool])
            print(f"\n🐶 dogAgent: {answer}\n")

            # 5. 保存对话到记忆
            if memory and session_id:
                memory["conv_store"].add_message(user_id, session_id, "user", user_input)
                memory["conv_store"].add_message(user_id, session_id, "assistant", answer)
                memory["session_mgr"].touch_session(session_id)

                # 6-8. 后台任务（不阻塞用户）
                if bg_tasks:
                    bg_tasks.submit_preference_extract(
                        llm, memory["user_prefs"], user_id, session_id, user_input, answer
                    )
                    bg_tasks.submit_compaction(
                        memory.get("compaction"), user_id, session_id
                    )
                    bg_tasks.submit_title_generate(
                        llm, memory["session_mgr"], session_id
                    )
                    if debug_mode:
                        print("  ⚡ 后台任务已提交（偏好提取/压缩/标题）")

        except Exception as e:
            print(f"\n❌ 错误: {e}\n")


def _handle_command(cmd: str, memory: dict, user_id: str, current_session_id: str) -> str:
    """处理 / 开头的会话管理命令，返回当前 session_id"""
    cmd_lower = cmd.lower().strip()

    if cmd_lower == "/new":
        new_id = memory["session_mgr"].create_session(user_id)
        print(f"  📂 新会话已创建: {new_id[:8]}...")
        return new_id

    elif cmd_lower == "/sessions":
        sessions = memory["session_mgr"].list_sessions(user_id)
        if not sessions:
            print("  （无会话记录）")
        else:
            print("  📋 会话列表:")
            for s in sessions:
                marker = "→" if s["id"] == current_session_id else " "
                title = s.get("title") or "(未命名)"
                print(f"  {marker} {s['id'][:8]}  {title}  [{s['updated_at']}]")
        return current_session_id

    elif cmd_lower.startswith("/switch "):
        target = cmd.split(maxsplit=1)[1].strip()
        sessions = memory["session_mgr"].list_sessions(user_id)
        matched = [s for s in sessions if s["id"].startswith(target)]
        if len(matched) == 1:
            print(f"  📂 已切换到: {matched[0]['id'][:8]} ({matched[0].get('title', '')})")
            return matched[0]["id"]
        elif len(matched) > 1:
            print(f"  ⚠️  匹配到 {len(matched)} 个会话，请输入更精确的 ID")
        else:
            print(f"  ⚠️  未找到匹配的会话")
        return current_session_id

    elif cmd_lower == "/prefs":
        prefs = memory["user_prefs"].get_active(user_id)
        if not prefs:
            print("  （无偏好记录）")
        else:
            print("  ⚙️  用户偏好:")
            for p in prefs:
                print(f"    - {p['content']}")
        return current_session_id

    elif cmd_lower.startswith("/pref "):
        # /pref 内容 手动添加偏好
        content = cmd.split(maxsplit=1)[1].strip()
        if content:
            memory["user_prefs"].add_preference(user_id, content)
            print(f"  ✅ 偏好已添加: {content}")
        else:
            print("  用法: /pref 偏好内容（如：我家狗叫旺旺）")
        return current_session_id

    elif cmd_lower == "/help":
        print("  📖 命令:")
        print("    /new         创建新会话")
        print("    /sessions    列出所有会话")
        print("    /switch ID   切换会话")
        print("    /prefs       查看偏好")
        print("    /pref 内容   添加偏好")
        print("    /help        显示帮助")
        return current_session_id

    else:
        print(f"  未知命令: {cmd}，输入 /help 查看帮助")
        return current_session_id


def single_query(question: str) -> str:
    """单次查询模式（无记忆，兼容旧版）"""
    llm = create_llm()
    retriever = create_retriever()

    results = retriever.retrieve(question, top_k=3)
    context = retriever.format_context(results)

    system_msg = SystemMessage(content=SYSTEM_PROMPT)
    context_msg = HumanMessage(
        content=CONTEXT_TEMPLATE.format(context=context) + f"\n\n用户问题: {question}"
    )

    response = llm.invoke([system_msg, context_msg])
    return response.content


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        # 命令行传入问题，单次回答
        question = " ".join(args)
        answer = single_query(question)
        print(answer)
    else:
        # 交互模式
        chat_loop()