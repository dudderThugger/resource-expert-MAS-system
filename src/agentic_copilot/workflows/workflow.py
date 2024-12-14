from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step

from agentic_copilot.models.agents.orchestration.orchestrator_agent import (
    OrchestratorAgent,
)
from agentic_copilot.models.utils.agents_util import AgentsState
from agentic_copilot.workflows.events import (
    CheckSuccesfulEvent,
    FinalResponseEvent,
    NeedInputEvent,
)
from agentic_copilot.workflows.generate_response import (
    generate_request_input,
    generate_response,
)
from agentic_copilot.workflows.utterance_checker import UtteranceChecker


class CopilotFlow(Workflow):

    @step
    async def run_checks(self, ev: StartEvent) -> CheckSuccesfulEvent | FinalResponseEvent:
        """Checks utterance if it asks anything that shouldn't be answered"""
        utterance = ev.utterance
        state = ev.state
        continue_bool = ev.continue_bool

        if continue_bool:
            check = True
            reasoning = "The conversation was continued no check needed"
        else:
            check, reasoning = await UtteranceChecker().check_utterance_async(utterance)

        if check:
            return CheckSuccesfulEvent(utterance=utterance, state=state, conversation_going=continue_bool)

        else:
            return FinalResponseEvent(message=reasoning, state=state)

    @step
    async def generate_completion(self, ev: CheckSuccesfulEvent) -> FinalResponseEvent | NeedInputEvent:
        """This step calls the orchestrator agent to generate a completion for the user's utterance"""
        utterance: str = ev.utterance
        state: AgentsState = ev.state
        conversation_going: bool = ev.conversation_going

        orchestrator_agent = OrchestratorAgent(state=state, continue_conversation=conversation_going)
        state.chat_history.append(f"User to Orchestrator agent: {utterance}.")
        status, message = await orchestrator_agent.achat(utterance)

        if status == OrchestratorAgent.EXECUTION_DONE:
            return FinalResponseEvent(message=message, state=state)

        elif status == OrchestratorAgent.NEED_INPUT:
            return NeedInputEvent(message=message, state=state)

    @step
    async def generate_final_answer(self, ev: FinalResponseEvent) -> StopEvent:
        answer = await generate_response(state=ev.state)
        return StopEvent(result=(answer, ev.state))

    @step
    async def generate_need_input_answer(self, ev: NeedInputEvent) -> StopEvent:
        answer = await generate_request_input(state=ev.state, message=ev.message)
        return StopEvent(result=(answer, ev.state))
