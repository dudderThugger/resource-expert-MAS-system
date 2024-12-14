from llama_index.core.base.llms.types import ChatMessage
from pydantic import BaseModel


class AgentResponseState(BaseModel):
    chat_history: list[ChatMessage]
    plan: list[tuple[str, str]]
    current_step: int
    # data: DataFrame


class AgentRequest(BaseModel):
    utterance: str
    state: AgentResponseState


class AgentResponse(BaseModel):
    base_utterance: str
    answer: str
    state: AgentResponseState
