"""
dogAgent LLM 工厂（统一入口）

所有模块通过 create_llm() 获取 LLM 实例，支持两种后端：
  - qwen:   通义千问 (DashScope OpenAI 兼容接口) — 默认
  - openai:  OpenAI / 任何兼容 OpenAI 格式的 API

用法:
    from agent.llm import create_llm
    llm = create_llm()                          # 默认参数
    llm = create_llm(temperature=0.1, max_tokens=4096)  # 自定义参数
"""

import os
import logging

logger = logging.getLogger("llm")

CHAT_MODEL = os.environ.get("CHAT_MODEL", "qwen-plus")


def create_llm(temperature: float = 0.3, max_tokens: int = 2048):
    """
    创建 LLM 实例。

    根据 LLM_PROVIDER 环境变量选择后端:
      - qwen:   QWEN_API_KEY + DashScope
      - openai:  OPENAI_API_KEY + OPENAI_API_BASE
    """
    from langchain_openai import ChatOpenAI

    llm_provider = os.environ.get("LLM_PROVIDER", "qwen").lower()

    if llm_provider == "qwen":
        llm = ChatOpenAI(
            model=CHAT_MODEL or "qwen-plus",
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=os.environ.get("QWEN_API_KEY") or os.environ.get("OPENAI_API_KEY"),
            openai_api_base=os.environ.get("QWEN_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        )
        logger.info(f"LLM: Qwen / {CHAT_MODEL}")
    else:
        llm = ChatOpenAI(
            model=CHAT_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            openai_api_base=os.environ.get("OPENAI_API_BASE"),
        )
        logger.info(f"LLM: OpenAI / {CHAT_MODEL}")

    return llm
