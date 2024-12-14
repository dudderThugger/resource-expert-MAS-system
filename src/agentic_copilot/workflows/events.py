from llama_index.core.workflow import Event

from agentic_copilot.models.utils.agents_util import AgentsState


class AgentResponseEvent(Event):
    state: AgentsState


class CheckSuccesfulEvent(Event):
    utterance: str
    state: AgentsState | None
    conversation_going: bool


class NeedInputEvent(AgentResponseEvent):
    message: str


class FinalResponseEvent(AgentResponseEvent):
    message: str
