import sys

from llama_index.core import PromptTemplate
from llama_index.core.tools import FunctionTool

from agentic_copilot.models.agents.orchestration.planning_agent import PlanningAgent
from agentic_copilot.models.agents.orchestration.research_agent import ResearchAgent
from agentic_copilot.models.agents.query.calculation_agent import CalculationAgent
from agentic_copilot.models.agents.query.query_orchestrator_agent import (
    QueryOrchestratorAgent,
)
from agentic_copilot.models.utils.agent_base import AgentBase, AgentFrameWork
from agentic_copilot.models.utils.agents_util import (
    AgentsState,
    Speaker,
    eval_response,
    get_logger,
)
from agentic_copilot.models.utils.llm_utils import LLMModels


class OrchestratorAgent(AgentBase):
    id: Speaker = Speaker.ORCHESTRATOR
    EXECUTION_DONE = "EXECTUION_DONE"
    NEED_INPUT = "NEED_INPUT"

    orchestrator_template = PromptTemplate(
        """
        <system_prompt>
        YOU ARE A HIGHLY-ACCLAIMED **ORCHESTRATOR AGENT**, RENOWNED FOR YOUR EXPERTISE IN COORDINATING COMPLEX MULTI-AGENT SYSTEMS. YOUR RESPONSIBILITY IS TO MANAGE FOUR SPECIALIZED AGENTS: A PLANNING AGENT, A QUERY ORCHESTRATOR AGENT, A CALCULATION AGENT, AND A RESEARCH AGENT. IN ADDITION, YOU MUST HANDLE SCENARIOS WHERE ANY OF THESE AGENTS REQUEST ADDITIONAL USER INPUT, WHICH YOU WILL FORWARD PROMPTLY TO THE USER AND RETURN THE RESPONSE TO THE RESPECTIVE AGENT.

        IT IS ALSO YOUR RESPONSIBILITY TO USE THE USER INPUT TO MODIFY THE PLAN SO WITH THE MODIFIED PLAN IT WILL BE CLEAR FOR THE AGENT THAT ISSUED THE REQUEST.
        {{start_or_continue_conversation}}

        ### GOAL ###

        - YOUR MAIN OBJECTIVE IS TO SUCCESSFULLY ANSWER THE USER'S QUESTION BY DECOMPOSING IT INTO TASKS, COORDINATING INTERACTIONS AMONG THE AGENTS, AND HANDLING ANY REQUESTS FOR USER INPUT ALONG THE WAY.

        ### AGENT ROLES ###

        1. **PLANNING AGENT**: DESIGNS A DETAILED STRATEGY FOR SOLVING THE QUERY. THIS MAY INCLUDE IDENTIFYING DATA REQUIREMENTS, CALCULATION STEPS, OR REQUESTING USER INPUT TO COMPLETE THE PLAN.
        2. **QUERY ORCHESTRATOR AGENT**: EXECUTES QUERIES BASED ON THE PLAN TO GATHER ANY NECESSARY DATA, EITHER FROM INTERNAL SYSTEMS OR EXTERNAL SOURCES. THIS AGENT MAY ALSO REQUEST ADDITIONAL INPUT IF CRUCIAL DATA IS MISSING.
        3. **CALCULATION AGENT**: PERFORMS CALCULATIONS BASED ON THE RETRIEVED DATA TO GENERATE THE ANSWER. IF ANY DATA IS INCOMPLETE OR AMBIGUOUS, IT MAY ASK FOR USER CLARIFICATION.
        4. **RESEARCH AGENT**: GATHERS BACKGROUND INFORMATION AND EXTERNAL RESOURCES TO SUPPORT MORE COMPLEX OR KNOWLEDGE-INTENSIVE QUERIES. IF ADDITIONAL INPUT OR CLARIFICATION IS REQUIRED TO FOCUS THE RESEARCH, THIS AGENT MAY REQUEST MORE SPECIFIC INFORMATION FROM THE USER.

        ### INSTRUCTION ###

        1. **RECEIVE USER INPUT**:
            - WHEN A NEW QUESTION IS RECEIVED, FIRST FORWARD IT TO THE PLANNING AGENT. WAIT FOR THE PLANNING AGENT TO RETURN A COMPREHENSIVE PLAN THAT DETAILS DATA NEEDED, STEPS TO FOLLOW, AND ANY USER INPUT REQUIRED.
            - IF THE PLAN INDICATES THE NEED FOR BACKGROUND RESEARCH, ASSIGN THE TASK TO THE RESEARCH AGENT BEFORE PROCEEDING TO QUERY OR CALCULATION STEPS.
            - WHEN AN ANSWER IS PROVIDED FOR AN INPUT REQUEST, CONTINUE THE CONVERSATION WHERE YOU LEFT IT OFF.

        2. **USER INPUT HANDLING**:
            - IF ANY AGENT REQUESTS ADDITIONAL INPUT FROM THE USER (E.G., SPECIFYING MISSING DATA, PREFERENCES, ETC.), YOU MUST FORWARD THIS REQUEST TO THE USER IMMEDIATELY. DON'T CHANGE THE QUESTION; FORWARD IT AS IT WAS ASKED.
            - AFTER RECEIVING USER INPUT, USE THE CORRESPONDING TOOL TO MODIFY THE PLAN BASED ON THE INPUT. THEN EXECUTE THE STEP THAT WAS RETURNED BY THE TOOL.

        3. **EXECUTING THE PLAN**:
            - AFTER RECEIVING THE COMPLETE PLAN, START EXECUTING IT WITH YOUR TOOLS.
            - IF THE PLAN REQUIRES BACKGROUND RESEARCH, PASS THE TASK TO THE RESEARCH AGENT. WAIT FOR THE RESEARCH AGENT TO RETURN THE REQUIRED INFORMATION BEFORE PROCEEDING.
            - AFTER RESEARCH IS COMPLETED, PASS THE APPROPRIATE QUERIES TO THE QUERY ORCHESTRATOR AGENT.
            - IF MORE USER INPUT IS REQUIRED TO COMPLETE ANY QUERY PROCESS, REQUEST IT IMMEDIATELY AND RETURN THE RESPONSE TO THE QUERY ORCHESTRATOR AGENT.
            - IF CALCULATION IS REQUIRED AFTER DATA GATHERING, PASS THE INSTRUCTION TO THE CALCULATION AGENT FOR PROCESSING.
            - IF THE CALCULATION AGENT OR RESEARCH AGENT REQUIRES MORE INFORMATION (FROM THE USER OR OTHER SOURCES), FACILITATE THIS EXCHANGE.

        4. **RETURNING FINAL RESULT**:
            - AFTER ALL STEPS ARE EXECUTED, USE THE **done** TOOL TO RETURN AN ANSWER FOR THE USER.

        ### CHAIN OF THOUGHT ###

        1. **INITIAL ANALYSIS**:
            - FORMULATE A QUESTION BASED ON THE LAST MESSAGES AND THE ORIGINAL UTTERANCE.
            - FORWARD THE QUESTION TO THE PLANNING AGENT AND RECEIVE THE PLAN.
            - ENSURE THE PLAN IS COMPLETE, INCLUDING ALL REQUIRED STEPS AND DATA NEEDS.
            - IF THE PLANNING AGENT REQUESTS MORE USER INPUT, FORWARD THE REQUEST PROMPTLY TO THE USER AND AWAIT THEIR RESPONSE.

        2. **RESEARCH** (OPTIONAL):
            - IF THE PLAN REQUIRES BACKGROUND INFORMATION, ASSIGN THE TASK TO THE RESEARCH AGENT.
            - WAIT FOR THE RESEARCH AGENT TO RETURN THE INFORMATION BEFORE PROCEEDING TO QUERY OR CALCULATION STEPS.

        3. **DATA RETRIEVAL**:
            - AFTER THE PLAN (AND RESEARCH, IF NEEDED) IS FINALIZED, PASS THE APPROPRIATE QUERIES TO THE QUERY ORCHESTRATOR AGENT.
            - IF MORE USER INPUT IS REQUIRED TO COMPLETE THE QUERY PROCESS, REQUEST IT IMMEDIATELY AND RETURN THE RESPONSE TO THE QUERY ORCHESTRATOR AGENT.

        4. **CALCULATION**:
            - ONCE DATA IS GATHERED, SEND IT TO THE CALCULATION AGENT TO GENERATE THE FINAL ANSWER.
            - IF THE CALCULATION AGENT REQUIRES MORE USER INPUT, FORWARD THIS REQUEST, WAIT FOR THE USER'S RESPONSE, AND CONTINUE THE PROCESS.

        5. **FINAL RESPONSE**:
            - USE THE **done** TOOL TO RETURN TO THE USER; THE ANSWER WON'T BE FORMATTED BY YOU, BUT IN A DIFFERENT STEP.

        ### EDGE CASE HANDLING ###
        - IF THE PLANNING AGENT RETURNS AN INCOMPLETE OR UNCLEAR PLAN, REQUEST A REVISION UNTIL ALL NECESSARY DETAILS ARE INCLUDED.
        - IF ANY AGENT REQUESTS ADDITIONAL USER INPUT AND THE USER FAILS TO PROVIDE IT, RETURN TO THE USER AND EXPLAIN WHY THE INFORMATION IS NECESSARY TO COMPLETE THE TASK.
        - IF ANY DATA OR CALCULATION IS MISSING, INCOMPLETE, OR INCONSISTENT, RETURN TO THE RELEVANT AGENT AND REQUEST A CORRECTION BEFORE MOVING FORWARD.

        ### USER INPUT EXAMPLES ###
        - IF THE QUERY INVOLVES AMBIGUOUS OR MISSING DATA, FOR EXAMPLE: "Please specify the currency for revenue calculation," ENSURE THAT THE REQUEST IS PASSED TO THE USER AND WAIT FOR THEIR RESPONSE.
        - IF THE USER INPUT REQUIRED IS TOO COMPLEX OR UNAVAILABLE, ADVISE THE USER TO PROVIDE ALTERNATIVE INFORMATION OR SIMPLIFY THE QUERY.

        ### ANSWERING THE QUESTION ###
        - ONLY ANSWER WITH THE **done** TOOL.

        ### WHAT NOT TO DO ###
        - **NEVER** IGNORE REQUESTS FOR USER INPUT FROM ANY AGENT.
        - **DO NOT** PROVIDE THE USER WITH INCOMPLETE RESULTS; ENSURE ALL AGENT RESPONSES ARE FINAL AND ACCURATE BEFORE RETURNING AN ANSWER.
        - **NEVER** FAIL TO FORWARD USER INPUT PROMPTLY TO THE AGENT THAT REQUESTED IT.
        - **DO NOT** SKIP ANY AGENT INVOLVED IN THE PROCESS (PLANNING, QUERYING, RESEARCHING, OR CALCULATING).
        - **NEVER** ATTEMPT TO GENERATE A PLAN, QUERY DATA, PERFORM RESEARCH, OR DO CALCULATIONS YOURSELF—ALWAYS RELY ON THE DESIGNATED AGENTS.
        - **DO NOT** RETURN A FINAL ANSWER WITHOUT VERIFYING THAT ALL AGENT TASKS ARE COMPLETE.
        - **NEVER** RETURN ANSWER ON YOUR OWN; ALWAYS ANSWER WITH **need_input** OR **done** TOOLS.

        ### FEW-SHOT EXAMPLE ###
        {{few_shot_examples}}
        </system_prompt>

    """  # noqa: E501
    )

    start_conversation_template = PromptTemplate(
        """
        THE DETAILS OF A CONVERSATIONS ARE STORED IN AN OBJECT CALLED AgentsState. WHEN THE CONVERSATION REQUIRES USER INPUT WE CAN RESTORE THE AGENT BASED ON THIS STATE.
        NOW WE JUST STARTED THE CONVERSATION SO THIS STATE OBJECT IS IN ITS ORIGINAL STATE BYPASS THE ID OF THE STATE TO THE TOOLS THAT REQUIRE IT IN THER PARAMETER.

        ### STATE OF THE CONVERSATION ###
        {{agents_state}}
    """  # noqa: E501
    )

    continue_conversation_template = PromptTemplate(
        """
        YOU ALREADY STARTED THE CONVERSATION WITH THE USER BUT CONVERSATION STOPPED, BECAUSE YOU NEEDED SOME INPUT FROM THE USER OR YOU FINISHED THE TASK GIVEN.
        ACT BASED ON THE CHAT HISTORY AND THE NEXT INPUT.

        ### POSSIBLE ACTIONS ###
        - IF DURING THE EXECUTION OF THE PLAN USER INPUT WAS NEEDED, YOU HAVE TO MODIFY THE PLAN WITH THE modify_plan TOOL, EXECUTE THAT STEP FIRST BEFORE PROCEEDING TO THE NEXT ONE   !!!!THIS IS A LIKELY SCENARIO!!!
        - START A NEW PLAN IF THE PREVIOUS QUESTION WAS ALREADY ANSWERED AND THE NEXT INPUT IS A NEW QUESTION OR IF THE USER INPUT WAS FOR THE PLAN

        ### STATE OF THE CONVERSATION ###
        {{agents_state}}
    """  # noqa: E501
    )

    agent_tool_template = PromptTemplate(
        """
        Use this tool if you want to choose {agent_name} as the next speaker.
        The agent does the following:
        {agent_description}
        The instruction for the agent must be passed through the instruction parameter.
    """
    )

    few_shot_examples = [
        """
        **User Question**: "What will the total revenue be for Q4 if we sell the units at $50 per unit with a 10% expected return rate?"

        **Step 1**:
        - Pass the Question to the Planning Agent.
        **Planning Agent Output**:  ('PLAN_DONE', [('query_orchestrator_agent', 'Fetch how many units were sold in Q4.'),
                                                  ('calculation_agent', 'Calculate the revenue from the collected data with 50$ revenue per unit and 10% expected return rate.'),])

        **Step 2**:
        - Start executing the first step of the plan by calling proceed_to_next_step tool and then choose_query_orchestrator_agent tool with the instruction of the plan to the QueryOrchestratorAgent
        **Query Orchestrator Agent Output**: "(QUERY_NEED_INPUT, "I couldn't find a datastream for 'units sold'. Could you please provide more details or check the datastream name?")"

        **Step 3**
        - Pass the message of the orchestrator agent to the user with the need_input tool
        **User response**: "Oh sorry I meant the 'Production Line 4 - Units Produced' datastream"

        **Step 4**
        - Use the modify_plan tool with the modified instruction: "Fetch 'Production Line 4 - Units Produced' datastreams."
        **Tools response**: "(query_orchestrator_agent, \"Fetch 'Production Line 4 - Units Produced' datastreams.\")"

        **Step 5**
        - Execute the step by calling choose_query_orchestrator_agent tool with the instruction
        **Tools response**: "QUERY_DONE"

        **Step 5**:
        - Proceed to next step of plan by calling proceed_to_next_step tool and then choose_calculation_agent, send the data to the Calculation Agent.
        **Calculation Agent Output**: "CALCULATION_DONE"

        **Step 6**
        - Using the done tool to finish the plan and answer to the user
        **Final Response**: "EXECUTION_DONE"
        """,  # noqa: E501
    ]

    def __init__(
        self,
        continue_conversation: bool,
        state: AgentsState,
        model: LLMModels = LLMModels.GPT_4O,
        agent_framework: AgentFrameWork = AgentFrameWork.PROMPT,
        cli_print: bool = False,
    ) -> None:
        super().__init__(state=state, model=model, agent_framework=agent_framework, cli_print=cli_print)
        self.logger = get_logger(__name__, stream_output=sys.stdout)
        self.planning_agent = PlanningAgent(state=state)
        self.query_orchestrator_agent = QueryOrchestratorAgent(state=state)
        self.calculation_agent = CalculationAgent(state=state)
        self.research_agent = ResearchAgent(state=state)
        self.continue_conversation = continue_conversation

    def choose_planning_agent(self, instruction: str) -> str:
        """Use this tool to choose the planning agent as the next agent."""
        self.logger.info("Choose planning agent tool has been chosen")

        self.state.plan.clear()
        self.state.chat_history.append(f"Orchestrator agent to Planning agent: {instruction}")
        try:
            response = self.planning_agent.chat(instruction)
            response_status, response_body = eval_response(response)
        except Exception as e:
            message = f"Error occured when processing response of Planning agent: {str(e)}"
            self.logger.info(message)
            return message

        self.state.chat_history.append(f"Planning agent to Orchestrator agent: {str(response)}")

        if str(PlanningAgent.PLAN_DONE) == response_status:
            self.logger.info("Plan was finished, now proceed to execution.")
            self.state.current_step = 0
            return f"""
                Plan finished:
                {self.state.plan}

                Now proceed to the execution of the plan with calling the proceed to next step function.
            """

        elif str(PlanningAgent.PLAN_NEED_INPUT) == response_status:
            self.logger.info(f"Planning agent need input {response_body}")
            return f"""
                Planning agent need input to create the plan:
                {response_body}
            """

    def choose_query_orchestrator_agent(self, instruction: str) -> str:
        """ "Use this tool to choose the query orchestrator agent as the next agent"""
        self.logger.info("Choose Query Orcehstrator agent tool has been chosen")

        self.state.chat_history.append(f"Orchestrator agent to QueryOrchestrator agent: {instruction}")

        try:
            response = self.query_orchestrator_agent.chat(instruction)
            response_status, response_body = eval_response(response)
        except Exception as e:
            message = f"Error occured when processing response of QueryOrchestrator agent: {str(e)}"
            self.logger.info(message)
            return message

        self.state.chat_history.append(
            f"Query orchestrator agent to Orchestrator agent: {response_status, response_body}"
        )
        self.logger.info(f"Query Orcehstrator agent response {response}")

        if response_status == str(QueryOrchestratorAgent.QUERY_DONE):
            resp = """
                Query agent queried the required data and written in the state object.
                Now proceed with the execution by following the plan and execute the next step.
            """
            return resp

        if response_status == str(QueryOrchestratorAgent.QUERY_NEED_INPUT):
            self.logger.info(f"Unsuccesful query, need input: {response_body}")
            return f"""
                Query agent requires user input:
                {str(response)}
            """

    def choose_calculation_agent(self, instruction: str):
        """Use this tool to choose calculation agent as the next agent."""
        self.logger.info(f"Calculation agent tool has been chosen with instruction: {instruction}")

        self.state.chat_history.append(f"Orchestrator agent to Calculation agent: {instruction}")

        try:
            response = self.calculation_agent.chat(instruction)
            response_status, response_body = eval_response(response)

        except Exception as e:
            message = f"Error occured when processing response of CalculationAgent agent: {str(e)}"
            self.logger.info(message)
            return message

        self.state.chat_history.append(f"Calculation agent to Orchestrator agent: {str(response)}")
        self.logger.info(f"Calculation agent response: {str(response)}")

        if response_status == str(CalculationAgent.CALCULATION_DONE):
            return f"""
                Calculation successful value written in the state object:
                {self.state.calculation_results}
            """

        elif response_status == str(CalculationAgent.CALCULATION_NEED_INPUT):
            self.logger.info(f"Unsuccesful calculation, need input: {response_body}")
            return f"""
                Calculation need input: {response_body}
            """

    def choose_research_agent(self, instruction: str):
        """Use this tool to choose calculation agent as the next agent."""
        self.logger.info(f"Research agent tool has been chosen with instruction: {instruction}")

        self.state.chat_history.append(f"Orchestrator agent to Research agent: {instruction}")

        try:
            response = self.research_agent.chat(instruction)
            response_status, response_body = eval_response(response)

        except Exception as e:
            message = f"Error occured when processing response of CalculationAgent agent: {str(e)}"
            self.logger.info(message)
            return message

        self.state.chat_history.append(f"Calculation agent to Orchestrator agent: {str(response)}")
        self.logger.info(f"Calculation agent response: {str(response)}")

        if response_status == str(ResearchAgent.RESEARCH_DONE):
            return f"""
                Research was successful value written in the state object:
                {self.state.research_results}
            """

        elif response_status == str(ResearchAgent.RESEARCH_NEED_INPUT):
            self.logger.info(f"Research need input: {response_body}")
            return f"""
                Calculation need input: {response_body}
            """

    def need_input(self, question) -> tuple[str, str]:
        """Use this toon when you need to ask a question from the user or one of you agents requested a new input from
        the user. The question parameter must contain the input request adressed to the user.
        Always use this or the 'done' tool before returning to the user.
        """
        self.logger.info("Need input tool has been chosen")
        self.state.chat_history.append(f"Orchestrator agent to user: {question}.")

        return (self.NEED_INPUT, question)

    def done(self, message) -> tuple[str, str]:
        """Use this tool when you are finished with the execution of the plan.
        Format an answer for the user based on the plans result and the original question.
        Always use this tool or the 'need_input' tool before returning to the user.
        """
        self.logger.info("Execution finished tool has been chosen")
        self.state.chat_history.append(f"Orchestrator agent to user: {self.EXECUTION_DONE}.")

        return self.EXECUTION_DONE, message

    def chat(self, message: str) -> str:
        agent = self.agent_factory()
        self.state.chat_history.append(f"User to agent: {message}.")
        response = eval_response(agent.chat(message))

        return response.response

    async def achat(self, message: str) -> tuple[str, str]:
        agent = self.agent_factory()
        self.state.chat_history.append(f"User to agent: {message}.")
        response = eval_response((await agent.achat(message)).response)

        return response

    @property
    def agent_description(self) -> str:
        return """
        An orchestration agent coordinates multiple specialized agents to answer a user’s query by managing tasks such
        as querying user records, executing database queries, and retrieving information from company documents.
        """

    @property
    def system_prompt(self) -> str:
        if not self.continue_conversation:
            start_conversation_prompt = self.start_conversation_template.format(
                agents_state=self.state.get_state_string()
            )
            return self.orchestrator_template.format(
                start_or_continue_conversation=start_conversation_prompt,
                # few_shot_examples=self.few_shot_examples,
            )

        else:
            continue_converstion = self.continue_conversation_template.format(
                agents_state=self.state.get_state_string()
            )
            return self.orchestrator_template.format(
                start_or_continue_conversation=continue_converstion,
                # few_shot_examples=self.few_shot_examples,
            )

    @property
    def tools(self) -> list[FunctionTool]:
        return [
            FunctionTool.from_defaults(fn=self.need_input, name="need_input", return_direct=True),
            FunctionTool.from_defaults(fn=self.choose_planning_agent, name="choose_planning_agent"),
            FunctionTool.from_defaults(fn=self.choose_query_orchestrator_agent, name="choose_query_orchestrator_agent"),
            FunctionTool.from_defaults(fn=self.choose_calculation_agent, name="choose_calculation_agent"),
            FunctionTool.from_defaults(fn=self.done, name="done", return_direct=True),
        ]
