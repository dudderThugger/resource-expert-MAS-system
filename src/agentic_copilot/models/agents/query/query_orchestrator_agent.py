import sys

from llama_index.core import PromptTemplate
from llama_index.core.tools import FunctionTool

from agentic_copilot.models.agents.query.datastream_query_agent import (
    DataStreamQueryAgent,
)
from agentic_copilot.models.agents.query.invoice_query_agent import InvoiceQueryAgent
from agentic_copilot.models.utils.agent_base import AgentBase, AgentFrameWork, QueryAgentBase
from agentic_copilot.models.utils.agents_util import (
    AgentsState,
    Speaker,
    eval_response,
    get_logger,
)
from agentic_copilot.models.utils.llm_utils import LLMModels


class QueryOrchestratorAgent(AgentBase):
    id: str = Speaker.QUERY_ORCHESTRATOR.value

    QUERY_DONE = "QUERY_DONE"
    QUERY_NEED_INPUT = "QUERY_NEED_INPUT"

    prompt_template = PromptTemplate(
        """
        YOU ARE A SECOND-LEVEL QUERY ORCHESTRATOR AGENT WITHIN A HIERARCHICAL MULTI-AGENT SYSTEM. YOUR TASK IS TO DELEGATE THE GIVEN QUESTIONS TO THE CORRECT QUERY AGENT UNDER YOUR MANAGEMENT.

        ### STATE CONTEXT ###
        {state_representation}
        *PLEASE REVIEW THIS STATE INFORMATION CAREFULLY BEFORE DECIDING WHICH QUERY AGENT TO SELECT OR WHETHER TO REQUEST USER CLARIFICATION.*

        ### INSTRUCTIONS ###

        1. **ROLE**: YOU ARE RESPONSIBLE FOR SELECTING THE APPROPRIATE QUERY AGENT TO EXECUTE A TASK BASED ON THE QUESTION YOU RECEIVE FROM THE PRIMARY ORCHESTRATOR AGENT. **ALWAYS CONSIDER THE STATE CONTEXT ({{state_representation}}) TO DETERMINE ANY RELEVANT INFORMATION ABOUT PREVIOUS INTERACTIONS OR USER REQUESTS** BEFORE FORMING A QUERY.

        2. **AGENT SELECTION**:
            - YOU MUST CHOOSE THE CORRECT QUERY AGENT BASED ON THE GIVEN QUESTION AND THEN UTILIZE THE CORRESPONDING TOOL TO ADD THAT AGENT TO YOUR PLAN.
            - IF THE QUESTION FALLS OUTSIDE THE SCOPE OF ANY OF YOUR QUERY AGENTS, DO NOT ATTEMPT TO DELEGATE THE QUESTION. INSTEAD, REQUEST CLARIFICATION FROM THE USER WITH THE 'need_input' TOOL BY ASKING IF THEY ARE CERTAIN ABOUT THEIR QUESTION.
            - EVEN IF YOU THINK THE QUERY IS UNCERTAIN, DELEGATE THE QUESTION. LET THE SPECIFIC AGENT ASK QUESTIONS FROM THE USER AS NEEDED.

        3. **CONSIDERING STATE REPRESENTATION**:
            - USE THE INFORMATION PROVIDED IN **STATE CONTEXT** TO ASSESS IF THERE IS ANY RELEVANT CONTEXT OR PRIOR DETAILS THAT MIGHT IMPACT YOUR AGENT SELECTION OR INSTRUCTIONS.
            - **IF THE STATE CONTEXT INDICATES THAT CLARIFICATION WAS PREVIOUSLY REQUESTED OR THAT THERE IS AMBIGUITY IN THE USER'S QUESTION, PRIORITIZE REQUESTING CLARIFICATION** WITH `need_input` IF NEEDED.
            - **IF THE STATE INDICATES PREVIOUS AGENT SELECTION OR PARTIAL DATA RETRIEVAL, ENSURE THAT YOUR DECISION BUILDS ON THIS EXISTING CONTEXT** RATHER THAN DUPLICATING EFFORTS.

        ### TOOLS AVAILABLE ###

        - **`choose_<AGENT_NAME>`**: WHERE <AGENT_NAME> IS THE NAME OF THE AGENT.
            - USE THIS TOOL TO ASSIGN A TASK TO QUERY AGENT <AGENT_NAME>. USE THIS WHEN THE QUESTION FALLS WITHIN THE SCOPE OF AGENT <AGENT_NAME>'S EXPERTISE.
        - **`done`**: USE THIS TOOL TO FINALIZE YOUR RESPONSE AFTER DELEGATING A QUESTION OR DETERMINING THAT FURTHER INPUT IS NEEDED.
        - **`need_input`**: USE THIS TOOL TO ASK THE USER FOR ADDITIONAL CLARIFICATION IF THE QUESTION DOES NOT FIT INTO THE SCOPE OF ANY AVAILABLE QUERY AGENT.
            - ALSO USE THIS TOOL WHEN ONE OF THE TOOLS OUTPUT TOLD YOU TO. THE 'question_to_user' PARAMETER IS WHAT YOU CAN ASK QUESTION WITH-

        ### CHAIN OF THOUGHT TO HANDLE A QUERY ###

        1. **REVIEW STATE REPRESENTATION**: START BY ANALYZING THE STATE CONTEXT PROVIDED IN <STATE CONTEXT> TO IDENTIFY ANY PRIOR INFORMATION OR CONTEXTUAL DETAILS THAT MIGHT IMPACT YOUR QUERY DECISION.
        2. **UNDERSTAND THE QUESTION**: ANALYZE THE GIVEN QUESTION OR PROBLEM. IDENTIFY WHICH QUERY AGENT (IF ANY) CAN ADDRESS THE ISSUE EFFECTIVELY.
        3. **SELECT THE APPROPRIATE AGENT OR ASK FOR INPUT**:
            - IF THE QUESTION FALLS WITHIN THE SCOPE OF ONE OF YOUR QUERY AGENTS, USE THE CORRESPONDING `choose_X` TOOL TO ASSIGN THE TASK.
            - IF THE QUESTION DOES NOT ALIGN WITH ANY OF THE QUERY AGENTS, USE THE `need_input` TOOL TO ASK FOR CLARIFICATION FROM THE USER.
        4. **FINALIZE RESPONSE**: EITHER USE THE `done` TOOL TO CONFIRM THE ACTION YOU TOOK AND FORWARD IT DIRECTLY OR USE 'need_input' TO FORWARD THE AGENT'S QUESTION TO THE ORCHESTRATOR AGENT ABOVE YOU.

        ### QUERY AGENT DESCRIPTIONS ###
        {agent_descriptions}

        ### WHAT NOT TO DO ###

        - **NEVER ASSIGN A QUESTION TO AN AGENT THAT DOES NOT HAVE EXPERTISE IN THE GIVEN TOPIC.**
        - **NEVER PROVIDE A RESPONSE OTHER THAN THE FINALIZED OUTPUT USING `done` OR REQUESTING CLARIFICATION OR FURTHER INPUT USING `need_input`.**
        - **NEVER ATTEMPT TO ACCESS OR GENERATE DATA YOURSELF; ALWAYS DELEGATE TO AN AGENT OR ASK FOR CLARIFICATION.**
        - **NEVER REPHRASE THE QUESTIONS WHEN YOU HAVE TO FORWARD A NEED INPUT CALL**
        - **NEVER ANSWER WITH THE `done` TOOL IF THE QUERY WAS NOT RESOLVED YET**
        - **NEVER USE THE `need_input` TOOL WHEN YOU KNOW WHICH QUERY AGENT TO DELEGATE THE ANSWER TO**
        - **NEVER ASK FOR SPECIFICATIONS ABOUT THE DETAILS OF THE QUESTION LIKE SITES, ETC. LET YOUR QUERY AGENTS FIND OUT THE SPECIFICITIES THEY NEED.**

        ### FEW-SHOT EXAMPLES ###

        {few_shot_examples}

        ### SUMMARY ###

        YOUR ROLE IS TO DELEGATE QUESTIONS TO THE APPROPRIATE QUERY AGENTS BASED ON THEIR EXPERTISE USING THE TOOLS PROVIDED. ALWAYS TAKE INTO ACCOUNT THE STATE CONTEXT REPRESENTED IN **{{state_representation}}** BEFORE FORMING ANY QUERIES OR REQUESTING CLARIFICATION. FOLLOW THE CHAIN OF THOUGHT FOR OPTIMAL HANDLING OF QUERIES AND ENSURE THAT FINALIZATION IS PERFORMED WITH `done`.

        """  # noqa: E501
    )

    few_shot_examples = """
        #### Example 1 ####
        **State Representation**: "User previously asked for average energy consumption data by region."
        **Query**: "What was the average energy consumption in New York for January?"

        1. **REVIEW STATE REPRESENTATION**: THE STATE INDICATES THE USER IS FOCUSING ON ENERGY CONSUMPTION DATA BY REGION.
        2. **UNDERSTAND THE QUESTION**: THE QUESTION IS ENERGY-RELATED, SO IT FALLS WITHIN THE SCOPE OF DATASTREAM QUERY AGENT.
        3. **ASSIGN TO AGENT**: USE `choose_datastream_query_agent` WITH INSTRUCTION: "Fetch the average energy consumption in New York for January."
        4. **FINALIZE RESPONSE**: USE `done` TO OUTPUT THE RESPONSE.

        #### Example 2 ####
        **State Representation**: "Previous clarification requested about location."
        **Query**: "What is the state of my invoices?"

        1. **REVIEW STATE REPRESENTATION**: THE STATE INDICATES THAT THE USER WAS ASKED TO SPECIFY A LOCATION IN A PREVIOUS QUERY.
        2. **UNDERSTAND THE QUESTION**: INVOICES RELATE TO THE INVOICE QUERY AGENT, BUT A LOCATION MAY STILL BE REQUIRED.
        3. **ASK FOR CLARIFICATION**: USE `need_input` TO ASK THE USER, "Could you specify the location for the invoice records?"
        4. **FINALIZE RESPONSE**: WAIT FOR USER RESPONSE TO CONTINUE.
    """  # noqa: E501

    def __init__(
        self,
        state: AgentsState,
        model: LLMModels = LLMModels.GPT_4O,
        agent_framework: AgentFrameWork = AgentFrameWork.PROMPT,
        cli_print: bool = False,
    ) -> None:
        super().__init__(state=state, model=model, agent_framework=agent_framework, cli_print=cli_print)
        self.logger = get_logger(name=__name__, stream_output=sys.stdout)
        self.query_agents: dict[str, QueryAgentBase] = {
            DataStreamQueryAgent.id: DataStreamQueryAgent(state=self.state),
            InvoiceQueryAgent.id: InvoiceQueryAgent(state=self.state),
        }

    def need_input(self, question_to_user: str) -> tuple[str, str]:
        """Use this tool when you need further input to answer question. For instance when you can't decide which agent
        to delegate the question to or when one of the agents returned with need input.
        """

        ret_value = str(self.QUERY_NEED_INPUT), question_to_user
        self.state.chat_history.append(f"QueryOrchestrator agent to Orchestrator agent: {str(ret_value)}")

        return ret_value

    def choose_datastream_query_agent(self, instruction: str) -> str:
        """Choose this tool to delegate question to DataStreamQuery agent."""
        self.logger.info(f"Query for datastreams tool has been chosen! query: {instruction}")

        self.state.chat_history.append(f"QueryOrchestrator agent to DataStreamQueryAgent: {instruction}")
        response_status, response_message = eval_response(self.query_agents[DataStreamQueryAgent.id].chat(instruction))
        message = ""
        if response_status == str(DataStreamQueryAgent.DS_QUERY_DONE):
            message = f"""
                Query was succesful, the queried data is:
                    {self.state.queried_data}

                Now validate the result and call the done tool if the query is correct.
            """

        if response_status == str(DataStreamQueryAgent.DS_AGENT_NEED_INPUT):
            message = f"""
                Query was unsuccesful
                Now call the need_input tool to make the question for the user with the following question:
                {response_message}
            """

        self.logger.info(f"{message}: {response_message}")
        return message

    def choose_invoice_query_agent(self, instruction: str) -> str:
        """Choose this tool when you want to delegate a query for an invoice record"""
        self.logger.info(f"Choose invoice query agent tool has been chosen with instruction: {instruction}")

        self.state.chat_history.append(f"QueryOrchestrator agent to InvoiceQueryAgent: {instruction}")
        response_status, response_message = eval_response(self.query_agents[InvoiceQueryAgent.id].chat(instruction))
        message = ""
        if response_status == str(InvoiceQueryAgent.INVOICE_QUERY_DONE):
            message = f"""
                Query was succesful, the queried data is:
                    {self.state.queried_data}

                Now validate the result and call the done tool if the query is correct.
            """

        if response_status == str(InvoiceQueryAgent.INVOICE_QUERY_NEED_INPUT):
            message = f"""
                Query was unsuccesful
                Now call the need_input tool to make the question for the user as the following: {response_message}
            """

        self.logger.info(f"{message}: {response_message}")
        return message

    def done(self) -> str:
        """Use this tool when you are finished with the querying of the data and respond
        with the output of this function"""
        self.logger.info("Query done tool has been chosen")

        ret_value = self.QUERY_DONE
        self.state.chat_history.append(f"QueryOrchestrator agent to Orchestrator agent: {str(ret_value)}")

        return (str(self.QUERY_DONE), "")

    @property
    def tools(self):
        return [
            FunctionTool.from_defaults(fn=self.choose_datastream_query_agent, name="choose_datastream_query_agent"),
            FunctionTool.from_defaults(fn=self.choose_invoice_query_agent, name="choose_invoice_query_agent"),
            FunctionTool.from_defaults(fn=self.done, name="from_defaults", return_direct=True),
            FunctionTool.from_defaults(fn=self.need_input, name="need_input", return_direct=True),
        ]

    @property
    def system_prompt(self):
        query_agent_descriptions = "\n\n".join(
            [f"{key}:\n{value.query_description}" for key, value in self.query_agents.items()]
        )

        return self.prompt_template.format(
            state_representation=self.state.get_json(),
            agent_descriptions=query_agent_descriptions,
            few_shot_examples=self.few_shot_examples,
        )

    @property
    def agent_description(self) -> str:
        return """
        This agent routes queries to the appropriate query agents:
        1.	Invoice Query Agent: Handles queries about invoices, including their status, unique invoice names,
            and related site details (country, state).
        2.	Datastream Query Agent: Processes and analyzes datastreams (HR, Energy, Emissions) to provide insights,
            trends, and performance metrics.

        It ensures queries are directed to the correct agent based on the data type.
        """
