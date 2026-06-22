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
MAIN_CHAT_MODEL = os.environ.get("MAIN_CHAT_MODEL", CHAT_MODEL)


def create_llm(temperature: float = 0.3, max_tokens: int = 2048):
    """轻量任务用 LLM（query rewrite、标题生成、偏好提取等）"""
    return _build_llm(CHAT_MODEL, temperature, max_tokens)


def create_main_llm(temperature: float = 0.3, max_tokens: int = 4096):
    """主对话 LLM，使用 MAIN_CHAT_MODEL（默认 qwen-max，工具调用更可靠）"""
    return _build_llm(MAIN_CHAT_MODEL, temperature, max_tokens)


def _build_llm(model: str, temperature: float, max_tokens: int):
    """内部：根据 model 名创建 LLM 实例"""
    from langchain_openai import ChatOpenAI

    llm_provider = os.environ.get("LLM_PROVIDER", "qwen").lower()

    if llm_provider == "qwen":
        llm = ChatOpenAI(
            model=model or "qwen-plus",
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=os.environ.get("QWEN_API_KEY") or os.environ.get("OPENAI_API_KEY"),
            openai_api_base=os.environ.get("QWEN_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        )
        logger.info(f"LLM: Qwen / {model}")
    else:
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            openai_api_base=os.environ.get("OPENAI_API_BASE"),
        )
        logger.info(f"LLM: OpenAI / {model}")

    return llm


def create_embeddings():
    """文章向量化用 embedding 模型（text-embedding-v3 via DashScope）"""
    from langchain_openai import OpenAIEmbeddings

    llm_provider = os.environ.get("LLM_PROVIDER", "qwen").lower()

    if llm_provider == "qwen":
        return OpenAIEmbeddings(
            model="text-embedding-v3",
            openai_api_key=os.environ.get("QWEN_API_KEY") or os.environ.get("OPENAI_API_KEY"),
            openai_api_base=os.environ.get("QWEN_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        )
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
    )


def create_vision_llm(temperature: float = 0.3, max_tokens: int = 2048):
    """
    创建视觉 LLM 实例（用于分析用户上传的图片）。

    Qwen: qwen-vl-plus（支持图片输入）
    OpenAI: gpt-4o（原生多模态）
    """
    from langchain_openai import ChatOpenAI

    llm_provider = os.environ.get("LLM_PROVIDER", "qwen").lower()
    vision_model = os.environ.get("VISION_MODEL")

    if llm_provider == "qwen":
        model = vision_model or "qwen-vl-plus"
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=os.environ.get("QWEN_API_KEY") or os.environ.get("OPENAI_API_KEY"),
            openai_api_base=os.environ.get("QWEN_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        )
        logger.info(f"Vision LLM: Qwen / {model}")
    else:
        model = vision_model or "gpt-4o"
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            openai_api_base=os.environ.get("OPENAI_API_BASE"),
        )
        logger.info(f"Vision LLM: OpenAI / {model}")

    return llm
