import time
from pathlib import Path
from typing import Callable

from agentic_copilot.models.agents.orchestration.orchestrator_agent import (
    OrchestratorAgent,
)
from agentic_copilot.models.agents.orchestration.planning_agent import PlanningAgent
from agentic_copilot.models.agents.orchestration.research_agent import ResearchAgent
from agentic_copilot.models.agents.query.calculation_agent import CalculationAgent
from agentic_copilot.models.agents.query.datastream_query_agent import (
    DataStreamQueryAgent,
)
from agentic_copilot.models.agents.query.invoice_query_agent import InvoiceQueryAgent
from agentic_copilot.models.agents.query.query_orchestrator_agent import (
    QueryOrchestratorAgent,
)
from agentic_copilot.models.utils.agent_base import AgentBase, AgentFrameWork
from agentic_copilot.models.utils.agents_util import (
    AgentsState,
    Speaker,
    load_state_from_json,
)
from agentic_copilot.models.utils.llm_utils import LLMModels
from tests.testing_utils import check_calculations, check_plan, check_queried_datas, check_researches, check_response

TEST_CASE_RUNS = 10


# Type annotations to convert string values
agents: dict[str, AgentBase] = {
    Speaker.CALCULATION: CalculationAgent,
    Speaker.DATASTREAM_QUERY: DataStreamQueryAgent,
    Speaker.INVOICE_QUERY: InvoiceQueryAgent,
    Speaker.ORCHESTRATOR: OrchestratorAgent,
    Speaker.PLANNING: PlanningAgent,
    Speaker.QUERY_ORCHESTRATOR: QueryOrchestratorAgent,
    Speaker.RESEARCH_AGENT: ResearchAgent,
}


async def research_metric(state: AgentsState, expected_state: AgentsState) -> bool:
    if not await check_researches(state.research_results, state.research_results):
        raise ValueError("Researches differ")

    return True


async def calculation_metrtic(state: AgentsState, expected_state: AgentsState) -> bool:
    if not await check_calculations(state.calculation_results, expected_state.calculation_results):
        raise ValueError("Calculations differ")

    return True


async def query_metrics(state: AgentsState, expected_state: AgentsState) -> bool:
    if not await check_queried_datas(state.queried_data, expected_state.queried_data):
        raise ValueError("One of the queries differ")

    return True


async def planning_metric(state: AgentsState, expected_state: AgentsState) -> bool:
    if not await check_plan(state.plan, expected_state.plan):
        raise ValueError("Plan differ")

    return True


async def orchestration_metric(state: AgentsState, expected_state: AgentsState) -> bool:
    return (
        planning_metric(state=state, expected_state=expected_state)
        and query_metrics(state=state, expected_state=expected_state)
        and calculation_metrtic(state=state, expected_state=expected_state)
    )


agents_metrics: dict[Callable[[AgentsState, AgentsState], bool]] = {
    Speaker.CALCULATION: calculation_metrtic,
    Speaker.DATASTREAM_QUERY: query_metrics,
    Speaker.INVOICE_QUERY: query_metrics,
    Speaker.ORCHESTRATOR: orchestration_metric,
    Speaker.PLANNING: planning_metric,
    Speaker.QUERY_ORCHESTRATOR: query_metrics,
    Speaker.RESEARCH_AGENT: research_metric,
}


async def aevaluate_test_result(
    agent_speaker: Speaker,
    response: tuple[str, str],
    expected_response: tuple[str, str],
    output_state: AgentsState,
    expected_output_state: AgentsState,
):
    if not await check_response(response, expected_response):
        return False
    return await agents_metrics[agent_speaker](output_state, expected_output_state)


async def arun_single_test(
    id: int,
    agent_speaker: Speaker,
    model: LLMModels,
    question: str,
    expected_response: tuple[str, str],
    input_state_path: str,
    expected_output_state_path: str,
    agent_framework: AgentFrameWork,
):
    state = load_state_from_json(Path(input_state_path))
    agent: AgentBase = agents[agent_speaker](
        state=state, model=LLMModels(model), agent_framework=agent_framework, cli_print=False
    )
    start_time = time.perf_counter()
    steps = 0
    reasoning = ""
    try:
        response = await agent.achat(question)
        end_time = time.perf_counter()
        expected_state = load_state_from_json(Path(expected_output_state_path))
        result = await aevaluate_test_result(
            agent_speaker=agent_speaker,
            response=response,
            expected_response=expected_response,
            output_state=state,
            expected_output_state=expected_state,
        )
        response_status = response[0]
        response_message = response[1]
    except Exception as e:
        result = False
        end_time = time.perf_counter()
        reasoning = {"exception_type": str(type(e)), "exception_message": str(e)}
        response_status = None
        response_message = None

    return {
        "id": id,
        "agent": agent_speaker,
        "model": model.value,
        "framework": agent_framework,
        "price": agent.tracer.price,
        "token": agent.tracer.input_tokens + agent.tracer.output_tokens,
        "time": (end_time - start_time),
        "result": 1 if result else 0,
        "steps": steps,
        "reasoning": reasoning,
        "response_status": response_status,
        "response_message": response_message,
    }
