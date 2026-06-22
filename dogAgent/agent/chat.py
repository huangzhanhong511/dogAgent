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

【网页搜索 / 图片搜索工具 — 必须使用！】
- 用户要图片/照片 → **立即调用 image_search**（英文关键词效果好，如 "miniature schnauzer dog"）
- 用户要链接/教程/产品/最新信息 → 调用 web_search
- 知识库内容不足时 → 调用 web_search
- 禁止说"我无法显示图片"或给用户手动搜索教程 — 直接调用工具

【拼多多商品搜索工具 — 严格规则】
- 用户问"在哪买"、"多少钱"、"推荐购买"、"哪款好"等购买意图 → **调用 pdd_search**
- 用户问"那些商品的详情"、"之前推荐的商品"、"继续查那些产品" → **逐一调用 pdd_search**，搜索上一轮回答中提到的每个品牌/商品名（如"Royal Canin"、"Hill's"等）
- 搜索时使用具体关键词（如"雪纳瑞低脂狗粮"、"Royal Canin 小型犬"）
- **展示规则（必须严格遵守）**：
  1. 工具返回的文本必须**原文复制**到回答中，不得改写商品名、不得修改价格、不得替换链接
  2. 链接中的 goods_id 是真实数据，**严禁自行生成或替换任何 goods_id**
  3. PDD API 只返回：商品名、价格、销量、店铺名、链接。没有配料表、脂肪含量、SGS证书
  4. **严禁对 API 未返回的字段作任何判断**——不得说某商品"含鸡油"、"无SGS报告"、"配料表有问题"等
  5. **营养成分数据严禁自行估算**：脂肪%、蛋白质%、热量等具体数值只能来自 wiki 知识库或 web_search 的检索结果；如无检索数据，必须说"具体营养成分请点击商品链接查看详情页"，**绝对不得引用训练记忆中的数字**
  6. 展示完工具原文后，可另起一段基于知识库给出选购提示（如"建议点开商品页核实营养成分表"）

【信息来源与可靠性追问 — 严格规则】
- 用户问"这信息怎么来的"、"可靠吗"、"你怎么知道"、"来源是什么"等 → **必须先回答来源**，再决定是否用工具验证：
  1. 商品名/价格/链接 → 来自拼多多开放 API（实时数据，价格和链接可信）
  2. 营养成分数值（脂肪%、蛋白质%等）→ 来自 AI 训练数据，**不保证准确**，建议点商品链接到详情页核实
  3. 犬只健康知识 → 来自 wiki 知识库或 web 搜索，注明来源
- 如需帮用户核实某条信息，可再次调用 pdd_search 或 web_search 进行 double check，并与之前结果对比说明差异
- 禁止跳过可靠性问题、直接用工具输出新商品列表当作"回答"

【图片分析（用户上传图片时）】
- 仔细描述图片内容，结合雪纳瑞专业知识分析
- 如有健康异常迹象，提示建议咨询兽医

回答时请：
1. 优先使用记忆中的用户信息回答个人相关问题
2. 使用知识库内容回答雪纳瑞专业问题
3. 需要图片时调用 image_search，需要链接/信息时调用 web_search，不要犹豫
4. 回答简洁友好
5. 涉及健康问题时，建议咨询兽医
"""

CONTEXT_TEMPLATE = """以下是根据用户问题检索到的相关知识库内容：

{context}

---
请基于以上内容回答用户的问题。如果内容不足以回答，请说明。
"""


def create_llm():
    """创建轻量 LLM 实例（委托给 agent.llm）"""
    try:
        from agent.llm import create_llm as _create
    except ImportError:
        from llm import create_llm as _create
    return _create()


def create_main_llm():
    """创建主对话 LLM 实例（委托给 agent.llm，使用 MAIN_CHAT_MODEL）"""
    try:
        from agent.llm import create_main_llm as _create
    except ImportError:
        from llm import create_main_llm as _create
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


def _get_ddgs():
    """返回 DDGS 实例，优先用新版 ddgs 包，回退到 duckduckgo_search。"""
    try:
        from ddgs import DDGS
        return DDGS()
    except ImportError:
        from duckduckgo_search import DDGS
        return DDGS()


def create_web_search_tool():
    """创建 web_search LangChain tool（文字信息搜索）。"""
    from langchain_core.tools import tool

    @tool
    def web_search(query: str) -> str:
        """搜索互联网获取文字信息。
        适用场景：
        - 知识库未覆盖的话题
        - 需要最新的价格、政策等实时信息
        - 用户要求推荐视频、教程、产品链接
        注意：获取图片时请使用 image_search 工具，不要用此工具。
        参数 query: 搜索关键词（中文或英文）
        """
        try:
            with _get_ddgs() as ddgs:
                results = list(ddgs.text(query, max_results=5))
            if not results:
                return "未找到相关搜索结果"
            lines = ["搜索结果：\n"]
            for r in results:
                title = r.get("title", "")
                url = r.get("href", "")
                snippet = r.get("body", "")
                lines.append(f"- **[{title}]({url})**\n  {snippet}")
            return "\n".join(lines)
        except Exception as e:
            return f"搜索出错: {e}"

    return web_search


def create_image_search_tool():
    """创建 image_search LangChain tool，返回结构化图片 URL 列表。"""
    from langchain_core.tools import tool

    @tool
    def image_search(query: str) -> str:
        """搜索图片，返回图片直链列表。
        适用场景：用户要求看某个话题的图片、照片。
        结果格式：JSON 字符串，包含 images 列表（每项含 image/thumbnail/title/url）。
        参数 query: 图片搜索关键词（英文效果更好，如 "miniature schnauzer dog"）
        """
        try:
            import json
            with _get_ddgs() as ddgs:
                results = list(ddgs.images(query, max_results=6))
            if not results:
                return json.dumps({"images": [], "summary": "未找到图片"})
            images = [
                {
                    "image": r.get("image", ""),
                    "thumbnail": r.get("thumbnail", ""),
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                }
                for r in results
                if r.get("image")
            ]
            return json.dumps({"images": images, "summary": f"找到 {len(images)} 张图片"}, ensure_ascii=False)
        except Exception as e:
            import json
            return json.dumps({"images": [], "summary": f"图片搜索出错: {e}"})

    return image_search


def invoke_with_tools(llm, messages: list, tools: list, max_iterations: int = 10) -> tuple[str, list[str]]:
    """ReAct 循环（LangGraph create_react_agent）。返回 (answer_text, image_urls)。"""
    import json
    from langchain_core.messages import AIMessage
    from langgraph.prebuilt import create_react_agent

    agent = create_react_agent(llm, tools)
    config = {"recursion_limit": max_iterations * 2 + 1}
    result = agent.invoke({"messages": messages}, config=config)

    # 最终回答：倒序找最后一条无 tool_calls 的 AIMessage
    answer = ""
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            content = msg.content
            answer = content if isinstance(content, str) else str(content)
            break

    # 图片 URL 从 image_search ToolMessage 中提取
    images: list[str] = []
    for msg in result["messages"]:
        if getattr(msg, "name", None) == "image_search":
            try:
                data = json.loads(msg.content)
                for img in data.get("images", []):
                    url = img.get("image") or img.get("thumbnail") or ""
                    if url:
                        images.append(url)
            except Exception:
                pass

    return answer, images


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
    llm = create_main_llm()
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
            img_tool = create_image_search_tool()
            search_tool = create_web_search_tool()
            try:
                from agent.pdd_tool import create_pdd_search_tool
            except ImportError:
                from pdd_tool import create_pdd_search_tool
            pdd_tool = create_pdd_search_tool()
            if memory and session_id:
                expand_tool = create_memory_expand_tool(memory, user_id)
                answer, _ = invoke_with_tools(llm, messages, [expand_tool, search_tool, img_tool, pdd_tool])
            else:
                answer, _ = invoke_with_tools(llm, messages, [search_tool, img_tool, pdd_tool])
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