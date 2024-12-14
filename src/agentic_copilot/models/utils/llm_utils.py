import json
from enum import Enum
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.openai_like import OpenAILike

import requests

from agentic_copilot.config import settings

litellm_proxy_base = "http://0.0.0.0:4000"


class LLMModels(str, Enum):
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet"
    CLAUDE_3_5_HAIKU = "claude-3-5-haiku"
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    # O1_MINI = "o1-mini"
    # O1_PREVIEW = "o1-preview"
    # LLAMA_GROQ_3_2_3B = "lllama-3-2-3b"
    LLAMA_GROQ_3_70B = "llama3-groq-70b"
    MIXTRAL_8X7B = "mixtral-8x7b"
    GEMMA_2_9B = "gemma2-9b"


def get_llm_prices() -> dict:
    response = requests.get(f"{litellm_proxy_base}/v1/model/info")
    dict = json.loads(response.content.decode("utf-8"))
    return {
        model["model_name"]: {
            "output_price": model["model_info"]["output_price"],
            "input_price": model["model_info"]["input_price"],
        }
        for model in dict["data"]
    }


def llm_factory_function(model, temperature: float = 0.0) -> OpenAILike:
    return OpenAILike(
        model=model,
        temperature=temperature,
        api_base=litellm_proxy_base,
        api_key="fake",
        is_function_calling_model=True,
        is_chat_model=True,
    )


def embedding_factory_function(
    model: str = settings.embedding_model,
    embedding_deployment_name: str = settings.embedding_deployment,
):
    return AzureOpenAIEmbedding(
        model=model,
        azure_endpoint=settings.azure_endpoint,
        deployment_name=embedding_deployment_name,
        api_key=settings.azure_api_key,
        temperature=settings.azure_temperature,
        api_version=settings.embedding_api_version,
    )


llm_prices = get_llm_prices()
