import sys
from typing import TextIO

from llama_index.core import PromptTemplate
from llama_index.core.tools import FunctionTool
from agentic_copilot.models.agents.orchestration.research_agent import ResearchAgent
from agentic_copilot.models.agents.query.calculation_agent import CalculationAgent
from agentic_copilot.models.agents.query.query_orchestrator_agent import (
    QueryOrchestratorAgent,
)
from agentic_copilot.models.utils.agent_base import AgentBase, AgentFrameWork
from agentic_copilot.models.utils.agents_util import (
    AgentsState,
    Speaker,
    get_logger,
)
from agentic_copilot.models.utils.llm_utils import LLMModels


class PlanningAgent(AgentBase):
    PLAN_DONE = "PLAN_DONE"
    PLAN_NEED_INPUT = "PLAN_NEED_INPUT"
    id: Speaker = Speaker.PLANNING

    prompt_template = PromptTemplate(
        """
        YOU ARE A PLANNING AGENT, TASKED WITH CREATING A PLAN TO ANSWER THE QUESTION(S) GIVEN TO YOU. YOUR OBJECTIVE IS TO CONSTRUCT A PLAN AS A LIST OF TUPLES THAT CONTAINS A SEQUENCE OF AGENTS TO BE EXECUTED IN THE SPECIFIC ORDER, ALONG WITH THE INSTRUCTIONS TO BE PROVIDED TO EACH AGENT. THE PLAN SHOULD GUIDE THE AGENTS EFFECTIVELY TO SOLVE THE QUERY PROVIDED.

        ### KEY RULES AND RESTRICTIONS ###

        - **ORCHESTRATION AND DATA ACCESS**: ONLY THE QUERY ORCHESTRATOR AGENT HAS ACCESS TO DATA OUTSIDE THE SCOPE OF THIS PLAN. ALL OTHER AGENTS DO NOT HAVE ACCESS TO ANY ADDITIONAL INFORMATION BESIDES WHAT THE QUERY ORCHESTRATOR FETCHES FOR THEM.
        - YOU MUST USE THE **TOOLS LISTED BELOW** TO ADD AGENTS TO THE PLAN.
        - ONCE THE PLAN IS FINALIZED, YOU MUST USE THE **`done` TOOL** TO FINALIZE IT AND RESPOND WITH ITS OUTPUT AND NOTHING ELSE. **USE THE `done` TOOL IF YOU HAVE COMPLETED THE PLAN**.

        ### TOOLS AVAILABLE ###

        - **`add_query_to_plan`**: USE THIS TOOL TO ADD A QUERY STEP TO THE PLAN. THIS REPRESENTS AN ACTION THAT INVOLVES FETCHING DATA OR INFORMATION VIA THE QUERY ORCHESTRATOR AGENT.
        - **add_research_to_plan**: USE THIS TOOL TO ADD A RESEARCH STEP TO THE PLAN. THIS REPRESENTS AN ACTION THAT INVOLVES FETCHING INFORMATION FROM THE COMPANY'S ESG DOCUMENT TO PROVIDE INSIGHTS ON SUSTAINABILITY EFFORTS, SOCIAL RESPONSIBILITY INITIATIVES, AND GOVERNANCE PRACTICES.
        - **`add_calculation_to_plan`**: USE THIS TOOL TO ADD A CALCULATION STEP TO THE PLAN. THIS IS USED WHEN THE TASK REQUIRES PERFORMING MATHEMATICAL OR LOGICAL OPERATIONS ON THE DATA COLLECTED.
        - **`done`**: USE THIS TOOL TO FINALIZE THE PLAN AND OUTPUT IT DIRECTLY. THIS SHOULD BE USED **ONLY AFTER ALL STEPS HAVE BEEN INCLUDED IN THE PLAN**.
        - **`start_planning`**: USE THIS TOOL TO INITIALIZE THE PLANNING PROCESS. THIS SHOULD BE USED AS THE FIRST STEP IN FORMULATING THE PLAN. GIVE IT THE INSTRUCTION YOU ARE CREATING THE PLAN FOR AS A PARAMETER.
        - **`need_input`**: USE THIS TOOL IF ADDITIONAL INPUT IS REQUIRED FROM THE USER.

        ### CHAIN OF THOUGHT TO CREATE A PLAN ###

        1. **UNDERSTAND THE QUESTION**: BEGIN BY UNDERSTANDING THE GIVEN QUESTION OR PROBLEM YOU NEED TO ADDRESS. DETERMINE WHAT INFORMATION IS REQUIRED AND WHETHER A QUERY IS NEEDED.
        2. **INITIALIZE PLANNING**: ALWAYS START BY USING THE `start_planning` TOOL TO INDICATE THE BEGINNING OF THE PLANNING PROCESS.
        3. **IDENTIFY REQUIRED ACTIONS**:
        - IF DATA COLLECTION IS REQUIRED, USE THE `add_query_to_plan` TOOL TO INCORPORATE THE QUERY ORCHESTRATOR AGENT INTO THE PLAN.
        - IF DATA RESEARCH IS REQUIRED, USE THE `add_research_to_plan` TOOL TO INCORPORATE THE RESEARCH AGENT INTO THE PLAN.
        - IF CALCULATIONS OR DATA PROCESSING ARE REQUIRED, USE THE `add_calculation_to_plan` TOOL TO ADD A CALCULATION STEP.
        4. **SEQUENCE THE AGENTS**: DETERMINE THE ORDER OF AGENTS AND MAKE SURE THEY ARE PLACED IN THE PLAN ACCORDING TO THE DEPENDENCIES OF THE TASK.
        5. **FINALIZE PLAN**: USE THE `done` TOOL TO FINALIZE THE PLAN AFTER ADDING ALL THE NECESSARY STEPS. ENSURE NO REDUNDANT ACTIONS ARE INCLUDED.

        ### WHAT NOT TO DO ###

        - **NEVER ATTEMPT TO ACCESS EXTERNAL DATA USING AGENTS THAT ARE NOT THE QUERY ORCHESTRATOR**. THE OTHER AGENTS DO NOT HAVE ACCESS TO ANY DATA BESIDES WHAT IS PROVIDED THROUGH THE PLAN.
        - **NEVER SKIP THE INITIALIZATION STEP** USING THE `start_planning` TOOL. THIS MUST ALWAYS BE INCLUDED AT THE START OF YOUR PLAN.
        - **NEVER OMIT THE FINALIZATION STEP**. ALWAYS USE THE `done` TOOL TO FINALIZE AND OUTPUT THE PLAN. !!!!!
        - **NEVER REQUEST ADDITIONAL INFORMATION UNLESS STRICTLY NECESSARY**. USE THE `need_input` TOOL ONLY WHEN THE PROVIDED QUESTION OR INSTRUCTION LACKS CLARITY.

        ### EXAMPLES OF PLAN FORMULATION ###

        #### Example 1 ####
        **Query**: "What is the average temperature in New York over the last week?"

        1. **START PLANNING**: Use `start_planning`.
        2. **QUERY DATA**: Use `add_query_to_plan` with instruction: `"Fetch the temperature data for New York from the last week."`
        3. **CALCULATE AVERAGE**: Use `add_calculation_to_plan` with instruction: `"Calculate the average temperature from the fetched data."`
        4. **FINALIZE PLAN**: Use `done` to output the complete plan.

        #### Example 2 ####
        **Query**: "How many users registered on our platform yesterday?"

        1. **START PLANNING**: Use `start_planning`.
        2. **QUERY DATA**: Use `add_query_to_plan` with instruction: `"Fetch the number of user registrations from yesterday."`
        3. **FINALIZE PLAN**: Use `done` to output the plan.

        ### FEW-SHOT EXAMPLES OF HOW A PLAN SHOULD LOOK LIKE ###
        {few_shot_examples}

        ### SUMMARY ###

        YOU MUST CREATE A PLAN CONSISTING OF A SEQUENCE OF TUPLES, REPRESENTING THE AGENTS TO BE EXECUTED AND THEIR INSTRUCTIONS, USING THE TOOLS PROVIDED. BE SURE TO FOLLOW THE RULES AND BEST PRACTICES MENTIONED ABOVE.
    """  # noqa: E501
    )

    agent_tool_template = PromptTemplate(
        """
        Use this tool if you want to add {agent_name} agent to the plan.
        The agent does the following:
        {agent_description}
        The instruction for the agent must be passed through the instruction parameter.
    """
    )

    few_shot_examples = [
        [
            (
                "query_orchestrator_agent",
                "Query the Data Center Energy Consumption data for the Colorado site",
            ),
            (
                "query_orchestrator_agent",
                "Query the Data Center Energy Consumption data for the Pennsylvanian site",
            ),
            (
                "calculation_agent",
                "Perform a regression analysis with the Pennsylvanian site's data as the independent variable and the Colorado site's data as the dependent variable",  # noqa: E501
            ),
        ],
        [
            (
                "query_orchestrator_agent",
                "Extract data for 'Return Products' from all sites for the years 2020-2024.",
            ),
            (
                "calculation_agent",
                "Calculate the mean of the extracted 'Return Products' data.",
            ),
        ],
    ]

    def __init__(
        self,
        state: AgentsState,
        model: LLMModels = LLMModels.GPT_4O,
        stream_output: TextIO = sys.stdout,
        agent_framework: AgentFrameWork = AgentFrameWork.PROMPT,
        cli_print: bool = False,
    ) -> None:
        super().__init__(
            state=state,
            model=model,
            agent_framework=agent_framework,
            cli_print=cli_print,
        )
        self.logger = get_logger(__name__, stream_output=stream_output)
        self.plan = []

    def start_planning(self, plan_utterance: str) -> None:
        """If there is no plan yet indicate with this tool that you started the planning process.
        Give the base utterance you are creating the plan for.
        You can restart the planning process with this tool if you feel something went wrong.
        """
        self.logger.info("start_planning tool has been chosen")
        self.state.plan.clear()
        self.state.base_utterance = plan_utterance
        return f"Planning started current plan:\n{self.plan}"

    def add_research_to_plan(self, instruction: str) -> str:
        self.logger.info("add_research_to_plan tool has been chosen")
        self.plan.append((ResearchAgent.id, instruction))
        return f"Research step added to plan, current plan: {self.plan}"

    def add_query_to_plan(self, instruction: str) -> str:
        self.logger.info("add_query_to_plan tool has been chosen")
        self.plan.append((QueryOrchestratorAgent.id, instruction))
        return f"Query step added to plan, current plan: {self.plan}"

    def add_calculation_to_plan(self, instruction: str) -> str:
        self.logger.info("add_calculation_to_plan tool has been chosen")
        self.plan.append((CalculationAgent.id, instruction))
        return f"Calculation step added to plan: {self.plan}"

    def need_input(self, question: str) -> tuple[str, str]:
        """Use this tool when you need further input for the plan
        your question must be exerted through the question parameter.
        Always use this tool before returning to the user"""
        self.logger.info("Need further input tool has been chosen")

        self.state.current_step = 0
        self.state.plan = self.plan

        ret_value = (str(self.PLAN_NEED_INPUT), question)

        return ret_value

    def done(self, message: str) -> str:
        """Use this tool when you finished the plan.
        Always use this tool or the 'need_input' tool before returning to the user."""
        self.logger.info(f"plan_done tool has been chosen plan: {self.plan}")

        self.state.current_step = 0
        self.state.plan = self.plan

        ret_value = (str(self.PLAN_DONE), message)

        return ret_value

    @property
    def tools(self) -> list[FunctionTool]:
        research_tool_description = self.agent_tool_template.format(
            agent_name=ResearchAgent.id, agent_description=str(ResearchAgent(self.state).agent_description)
        )
        query_tool_description = self.agent_tool_template.format(
            agent_name=QueryOrchestratorAgent.id,
            agent_description=str(QueryOrchestratorAgent(self.state).agent_description),
        )
        calculation_tool_description = self.agent_tool_template.format(
            agent_name=CalculationAgent.id,
            agent_description=str(CalculationAgent(self.state).agent_description),
        )
        return [
            FunctionTool.from_defaults(
                fn=self.add_research_to_plan, name="add_research_to_plan", description=research_tool_description
            ),
            FunctionTool.from_defaults(
                fn=self.add_query_to_plan, name="add_query_to_plan", description=query_tool_description
            ),
            FunctionTool.from_defaults(
                fn=self.add_calculation_to_plan,
                name="add_calculation_to_plan",
                description=calculation_tool_description,
            ),
            FunctionTool.from_defaults(fn=self.done, name="done", return_direct=True, description=self.done.__doc__),
            FunctionTool.from_defaults(
                fn=self.start_planning, name="start_planning", description=self.start_planning.__doc__
            ),
            FunctionTool.from_defaults(
                fn=self.need_input, name="need_input", return_direct=True, description=self.need_input.__doc__
            ),
        ]

    @property
    def system_prompt(self) -> str:
        return self.prompt_template.format(few_shot_examples=self.few_shot_examples)

    @property
    def agent_description(self) -> str:
        return """An agent that creates plans for an agentic system formulates a sequence of actions or strategies to
        achieve specific goals, considering available tools, constraints, and the system’s environment, ensuring
        optimal decision-making and adaptability."""
