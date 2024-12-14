from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, List

from llama_index.core import PromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.agent import FunctionCallingAgent, ReActAgent, AgentRunner
from llama_index.agent.lats import LATSAgentWorker
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms.function_calling import FunctionCallingLLM

from agentic_copilot.models.utils.agent_tracer import AgentTracer
from agentic_copilot.models.utils.agents_util import AgentsState, eval_response
from agentic_copilot.models.utils.llm_utils import (
    LLMModels,
    llm_factory_function,
)


from llama_index.core.tools import FunctionTool


class AgentFrameWork(str, Enum):
    BASE = "base"
    PROMPT = "prompt"
    REACT = "react"
    LATS = "lats"


BASE_TEMPLATE = PromptTemplate(
    """
You are a helpful assistant in a Multi-Agent system. Your job is to decide which tool to call based on user input. Use the available tool only when necessary. If unsure, ask clarifying questions.

### Rules:
- Do not guess inputs; ask the user for clarification if details are missing.
- Use the 'done' and 'need_input' tools before returning to the user.

## State with past coversations and variables you can use
{state_str}
"""  # noqa: E501
)

AGENT_PARAMS = {"max_function_calls": 10}
REACT_PARAMS = {"max_iterations": 20, "max_function_calls": 10}


def base_prompt_agent_factory(
    state_string: str,
    system_prompt: str,
    tools: List[FunctionTool],
    callback_manager: CallbackManager,
    model: LLMModels = LLMModels.GPT_4O,
):
    chat_history = [ChatMessage(role=MessageRole.SYSTEM, content=BASE_TEMPLATE.format(state_str=state_string))]
    return FunctionCallingAgent.from_tools(
        llm=llm_factory_function(model=model),
        tools=tools,
        chat_history=chat_history,
        callback_manager=callback_manager,
        **AGENT_PARAMS
    )


def function_calling_factory(
    state_string: str,
    system_prompt: str,
    tools: List[FunctionTool],
    callback_manager: CallbackManager,
    model: LLMModels = LLMModels.GPT_4O,
):
    return FunctionCallingAgent.from_tools(
        system_prompt=system_prompt,
        llm=llm_factory_function(model=model),
        tools=tools,
        callback_manager=callback_manager,
        **AGENT_PARAMS
    )


def react_factory(
    state_string: str,
    system_prompt: str,
    tools: List[FunctionTool],
    callback_manager: CallbackManager,
    model: LLMModels = LLMModels.GPT_4O,
):
    chat_history = [ChatMessage(role=MessageRole.SYSTEM, content=BASE_TEMPLATE.format(state_str=state_string))]
    return ReActAgent.from_tools(
        llm=llm_factory_function(model=model),
        tools=tools,
        callback_manager=callback_manager,
        chat_history=chat_history,
        **REACT_PARAMS
    )


def lats_factory(
    state_string: str,
    system_prompt: str,
    tools: List[FunctionTool],
    callback_manager: CallbackManager,
    model: LLMModels = LLMModels.GPT_4O,
):
    chat_history = [ChatMessage(role=MessageRole.SYSTEM, content=BASE_TEMPLATE.format(state_str=state_string))]
    return LATSAgentWorker.from_tools(
        llm=llm_factory_function(model=model), tools=tools, callback_manager=callback_manager, chat_history=chat_history
    ).as_agent()


FRAMEWORK_MAPPING: dict[AgentFrameWork, Callable[[str, List, LLMModels], AgentRunner]] = {
    AgentFrameWork.BASE: base_prompt_agent_factory,
    AgentFrameWork.PROMPT: function_calling_factory,
    AgentFrameWork.REACT: react_factory,
    AgentFrameWork.LATS: lats_factory,
}


class AgentBase(ABC):
    """Abstract base class for agents"""

    id: str

    def __init__(
        self,
        state: AgentsState,
        model: LLMModels,
        agent_framework: AgentFrameWork = AgentFrameWork.PROMPT,
        cli_print: bool = False,
    ):
        super().__init__()
        self.state = state
        self.model = model
        self.agent_framework = agent_framework
        self.cli_print = cli_print
        self.tracer = AgentTracer(model=model, cli_print=cli_print)
        self.callback_manager = CallbackManager(handlers=[self.tracer])
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.total_tokens = 0

    def chat(self, message: str) -> str:
        agent = self.agent_factory()
        response = agent.chat(message)

        return response.response

    async def achat(self, message: str) -> tuple[str, str]:
        agent = self.agent_factory()
        response = eval_response((await agent.achat(message)).response)

        return response

    @property
    @abstractmethod
    def agent_description(self) -> str:
        pass

    @property
    @abstractmethod
    def tools(self) -> list[FunctionTool]:
        pass

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        pass

    def agent_factory(self) -> AgentRunner:
        return FRAMEWORK_MAPPING[self.agent_framework](
            state_string=self.state.get_state_string(),
            system_prompt=self.system_prompt,
            tools=self.tools,
            model=self.model,
            callback_manager=self.callback_manager,
        )


class QueryAgentBase(AgentBase):
    @abstractmethod
    def __init__(self, state: AgentsState, model: LLMModels, agent_framework: AgentFrameWork, cli_print: bool) -> None:
        super().__init__(state=state, model=model, agent_framework=agent_framework, cli_print=cli_print)

    @property
    @abstractmethod
    def query_description(self) -> str:
        pass
