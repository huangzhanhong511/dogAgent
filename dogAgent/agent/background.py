"""
dogAgent 后台任务管理器

对话后的非阻塞工作统一入口：
- 偏好提取（LLM 自动从对话中提取用户偏好）
- DAG 压缩（摘要压缩检查）
- 会话标题生成

REDIS_URL 已设置 → Celery 模式：任务写入 Redis 队列，Worker 异步消费，重启不丢任务
REDIS_URL 未设置 → 线程池模式：进程内执行，适合本地开发
"""

import os
import json
import logging
from concurrent.futures import ThreadPoolExecutor, Future

logger = logging.getLogger("background")

# 默认配置
DEFAULT_MAX_WORKERS = 4
MAX_QUEUE_SIZE = 100

_REDIS_URL = os.environ.get("REDIS_URL")


class BackgroundTaskManager:
    """
    后台任务管理器。

    用法（chat.py）：
        bg = BackgroundTaskManager()
        bg.submit_preference_extract(llm, user_prefs, user_id, session_id, user_input, answer)
        bg.submit_compaction(compaction_engine, user_id, session_id)
        bg.submit_title_generate(llm, session_mgr, session_id)

    用法（server.py）：
        可直接用 FastAPI BackgroundTasks，或共享同一个 BackgroundTaskManager。

    退出时：
        bg.shutdown()
    """

    def __init__(self, max_workers: int = None):
        self._use_celery = bool(_REDIS_URL)
        if self._use_celery:
            logger.info("后台任务管理器已启动: Celery 模式 (Redis 队列，重启不丢任务)")
        else:
            workers = max_workers or int(os.environ.get("BG_MAX_WORKERS", str(DEFAULT_MAX_WORKERS)))
            self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="bg-task")
            self._pending: list[Future] = []
            logger.info(f"后台任务管理器已启动: 线程池模式 (max_workers={workers})")

    def submit_preference_extract(self, llm, user_prefs, user_id: str,
                                   session_id: str, user_input: str, answer: str) -> None:
        """提交偏好提取任务"""
        if self._use_celery:
            from agent.tasks import task_preference_extract
            task_preference_extract.delay(user_id, session_id, user_input, answer)
        else:
            self._submit(self._run_preference_extract, llm, user_prefs, user_id, session_id, user_input, answer)

    def submit_compaction(self, compaction_engine, user_id: str, session_id: str) -> None:
        """提交 DAG 压缩任务"""
        if compaction_engine is None:
            return
        if self._use_celery:
            from agent.tasks import task_compaction
            task_compaction.delay(user_id, session_id)
        else:
            self._submit(self._run_compaction, compaction_engine, user_id, session_id)

    def submit_title_generate(self, llm, session_mgr, session_id: str) -> None:
        """提交会话标题生成任务"""
        if self._use_celery:
            from agent.tasks import task_title_generate
            task_title_generate.delay(session_id)
        else:
            self._submit(self._run_title_generate, llm, session_mgr, session_id)

    def wait_all(self, timeout: float = 30):
        """等待所有挂起任务完成（线程池模式下有效）"""
        if self._use_celery:
            return
        done = []
        for f in self._pending:
            try:
                f.result(timeout=timeout)
            except Exception as e:
                logger.debug(f"后台任务异常: {e}")
            done.append(f)
        self._pending = [f for f in self._pending if f not in done]

    def shutdown(self):
        """关闭任务管理器"""
        if self._use_celery:
            return
        self.wait_all(timeout=10)
        self._executor.shutdown(wait=False, cancel_futures=True)
        logger.info("后台任务管理器已关闭")

    # ─── 内部方法（线程池模式） ───

    def _submit(self, fn, *args) -> Future | None:
        self._pending = [f for f in self._pending if not f.done()]
        if len(self._pending) >= MAX_QUEUE_SIZE:
            logger.warning(f"后台任务队列已满({MAX_QUEUE_SIZE})，跳过")
            return None
        future = self._executor.submit(fn, *args)
        self._pending.append(future)
        return future

    # ─── 任务实现 ───

    @staticmethod
    def _run_preference_extract(llm, user_prefs, user_id, session_id, user_input, answer):
        """后台任务：LLM 提取偏好"""
        try:
            from langchain_core.messages import HumanMessage

            conversation_text = f"用户: {user_input}\n助手: {answer}"
            prompt = user_prefs.build_extract_prompt(user_id, conversation_text)
            resp = llm.invoke([HumanMessage(content=prompt)])
            raw = resp.content if hasattr(resp, "content") else str(resp)
            raw = raw.strip()

            # 清理可能的 markdown 代码块包裹
            if raw.startswith("```"):
                import re
                raw = re.sub(r"^```\w*\n", "", raw)
                raw = re.sub(r"\n```$", "", raw)

            extraction = json.loads(raw)
            new_items = extraction.get("new", [])
            update_items = extraction.get("update", [])

            if new_items or update_items:
                user_prefs.apply_extraction(
                    user_id, extraction, source_session_id=session_id
                )
                logger.info(
                    f"偏好自动提取: user={user_id}, new={new_items}, update={len(update_items)} 条"
                )
            else:
                logger.debug(f"偏好提取: 无新偏好")

        except json.JSONDecodeError as e:
            logger.debug(f"偏好提取: LLM 返回非 JSON，跳过: {e}")
        except Exception as e:
            logger.debug(f"偏好提取失败: {e}")

    @staticmethod
    def _run_compaction(compaction_engine, user_id, session_id):
        """后台任务：DAG 压缩"""
        try:
            stats = compaction_engine.check_and_compact(user_id, session_id)
            if stats:
                logger.info(f"后台压缩完成: {stats}")
        except Exception as e:
            logger.debug(f"后台压缩失败: {e}")

    @staticmethod
    def _run_title_generate(llm, session_mgr, session_id):
        """后台任务：会话标题生成"""
        try:
            from langchain_core.messages import HumanMessage

            if not session_mgr.needs_title(session_id):
                return

            title_prompt = session_mgr.build_title_prompt(session_id)
            resp = llm.invoke([HumanMessage(content=title_prompt)])
            title = resp.content.strip()[:20] if hasattr(resp, "content") else ""
            if title:
                session_mgr.update_title(session_id, title)
                logger.info(f"后台标题生成: {session_id[:8]} → {title}")
        except Exception as e:
            logger.debug(f"后台标题生成失败: {e}")