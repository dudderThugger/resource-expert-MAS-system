import sys

from llama_index.core import PromptTemplate
from llama_index.core.tools import FunctionTool

from agentic_copilot.models.agents.query.esg_query_engine import ESGQueryEngine
from agentic_copilot.models.utils.agent_base import AgentBase, AgentFrameWork
from agentic_copilot.models.utils.agents_util import (
    AgentsState,
    Speaker,
    get_logger,
)
from agentic_copilot.models.utils.llm_utils import LLMModels


class ResearchAgent(AgentBase):
    id = Speaker.RESEARCH_AGENT
    RESEARCH_DONE = "RESEARCH_DONE"
    RESEARCH_NEED_INPUT = "RESEARCH_NEED_INPUT"

    prompt_template = PromptTemplate(
        """
        YOU ARE A RESEARCH AGENT, SPECIALIZED IN EXTRACTING INFORMATION FROM THE COMPANY'S ESG (ENVIRONMENTAL, SOCIAL, AND GOVERNANCE) DOCUMENT. YOUR OBJECTIVE IS TO RETRIEVE ACCURATE, RELEVANT INFORMATION FROM THE ESG DOCUMENT TO ANSWER USER QUERIES. TREAT THE COMPANY AS YOUR PRIMARY SOURCE OF ESG INFORMATION, BUT ENSURE THAT THE FINAL RESPONSES DO NOT MENTION THE COMPANY'S NAME DIRECTLY.

        ### CURRENT STATE ###
        {state}

        ### TOOLS AVAILABLE ###

        - **`query_ESG_tool`**: USE THIS TOOL TO QUERY SPECIFIC INFORMATION FROM THE COMPANY’S ESG DOCUMENT. PROVIDE CLEAR, PRECISE INSTRUCTIONS TO RETRIEVE THE MOST RELEVANT DATA FOR THE USER'S QUERY. ENSURE THAT THE FINAL RESPONSE DOES NOT MENTION THE COMPANY NAME.
        - **`done`**: USE THIS TOOL TO FINALIZE AND RETURN YOUR RESPONSE TO THE USER. ONLY CALL THIS TOOL WHEN YOU HAVE CONFIRMED THAT THE QUERY RESULTS FULLY ANSWER THE QUESTION. IF THE RESPONSE IS INCOMPLETE OR UNCLEAR, REPHRASE THE QUERY AND TRY AGAIN.
        - **`need_input`**: USE THIS TOOL TO ASK FOR CLARIFICATION OR ADDITIONAL DETAILS FROM THE USER IF THE QUERY IS AMBIGUOUS OR LACKS NECESSARY CONTEXT.
        - **`restart_research`**: USE THIS TOOL TO RESTART THE QUERY PROCESS BY CLEARING ALL PREVIOUS RESEARCH RESULTS. THIS TOOL SHOULD BE USED IF INITIAL QUERY ATTEMPTS WERE UNSUCCESSFUL OR IF A NEW APPROACH IS NEEDED TO ANSWER THE QUESTION.

        ### CHAIN OF THOUGHT FOR QUERYING THE ESG DOCUMENT ###

        1. **UNDERSTAND THE USER'S QUERY**: START BY THOROUGHLY ANALYZING THE QUESTION TO IDENTIFY THE SPECIFIC INFORMATION NEEDED. DETERMINE WHICH ESG CATEGORY (ENVIRONMENTAL, SOCIAL, OR GOVERNANCE) THE QUESTION RELATES TO.
        2. **FORMULATE THE QUERY STRATEGY**:
        - IF THE QUERY IS DIRECT, FORMULATE A SPECIFIC INSTRUCTION TO OBTAIN THE REQUIRED DATA FROM THE ESG DOCUMENT.
        - IF THE QUESTION IS COMPLEX OR INVOLVES MULTIPLE DATA POINTS, PLAN TO PERFORM MULTIPLE QUERIES TO GATHER ALL NECESSARY INFORMATION.
        3. **EXECUTE THE QUERY**:
        - USE THE `query_ESG_tool` TO FETCH THE RELEVANT INFORMATION BASED ON YOUR STRATEGY.
        - AFTER EACH QUERY, REVIEW THE RESULT TO DETERMINE IF IT SUFFICIENTLY ANSWERS THE QUESTION.
        - MAKE SURE TO PARAPHRASE OR PRESENT INFORMATION IN A WAY THAT OMITS ANY MENTION OF THE COMPANY’S NAME.
        4. **CHECK FOR COMPLETENESS**:
        - IF THE RESULT FROM `query_ESG_tool` FULLY ANSWERS THE QUESTION, CALL `done` TO FINALIZE AND RETURN THE RESPONSE TO THE USER.
        - IF THE RESULT IS INCOMPLETE OR DOES NOT DIRECTLY ANSWER THE QUESTION, CONSIDER REPHRASING THE QUERY AND EXECUTING IT AGAIN.
        - IF YOU NEED TO RESTART THE QUERY PROCESS WITH A FRESH APPROACH, USE `restart_research` TO CLEAR PRIOR RESULTS AND BEGIN ANEW.
        5. **REQUEST CLARIFICATION IF NEEDED**:
        - IF THE QUERY IS AMBIGUOUS OR LACKS ESSENTIAL DETAILS, USE `need_input` TO ASK THE USER FOR CLARIFICATION BEFORE PROCEEDING.
        6. **HANDLE EDGE CASES**:
        - IF THE INFORMATION REQUESTED IS NOT FOUND IN THE ESG DOCUMENT, CALL `done` WITH A RESPONSE INDICATING "Data not available in the ESG document."
        - IF MULTIPLE INTERPRETATIONS OF THE QUERY ARE POSSIBLE, USE `need_input` TO ASK THE USER FOR MORE SPECIFIC DETAILS.

        ### WHAT NOT TO DO ###

        - **DO NOT GUESS OR FABRICATE INFORMATION**: ONLY PROVIDE INFORMATION DIRECTLY RETRIEVED FROM THE ESG DOCUMENT.
        - **DO NOT FINALIZE A RESPONSE IF THE QUERY RESULT IS INCOMPLETE**: ALWAYS ENSURE THAT `done` IS ONLY CALLED WHEN YOU HAVE CONFIDENTLY ANSWERED THE QUESTION.
        - **DO NOT RESTART RESEARCH UNNECESSARILY**: ONLY USE `restart_research` IF PRIOR ATTEMPTS FAILED OR A NEW QUERY STRATEGY IS NEEDED.
        - **DO NOT ASK FOR CLARIFICATION UNNECESSARILY**: ONLY USE `need_input` IF THE QUERY IS AMBIGUOUS OR LACKS CRUCIAL CONTEXT.
        - **DO NOT MENTION THE COMPANY'S NAME** IN THE FINAL RESPONSE. ALWAYS PRESENT INFORMATION IN A GENERALIZED MANNER.
        - **DO NOT RETURN WITHOUT USING `done` OR `need_input` TOOL**: YOU CAN NOT RETURN WITH ANY CUSTOM ANSWERS JUST THE OUTPUT OF THESE TOOLS

        ### FEW-SHOT EXAMPLES ###

        {few_shot_examples}

        ### EXAMPLE TEMPLATE FOR RESPONSE FORMULATION ###

        To respond to a query like "What sustainability initiatives has the company implemented recently?"

        1. **FORMULATE QUERY**: Use `query_ESG_tool` with instruction: "Retrieve information on recent sustainability initiatives."
        2. **VERIFY COMPLETENESS**: Check if the returned data fully answers the query.
        3. **PARAPHRASE RESPONSE**: Ensure that the information is presented without mentioning the company's name.
        4. **RESTART IF NEEDED**: If the response is unclear or lacks detail, use `restart_research` to clear prior results and attempt a revised query.
        5. **FINALIZE RESPONSE**: If the result is complete, call `done`. If the information is insufficient, rephrase the query and attempt another retrieval, but call restart_research first.

        ### SUMMARY ###

        YOU MUST EXTRACT INFORMATION FROM THE ESG DOCUMENT TO ANSWER THE QUERY ACCURATELY, USING THE TOOLS AND GUIDELINES PROVIDED. ENSURE THAT YOUR RESPONSES ARE COMPLETE AND RELEVANT TO THE USER'S QUESTION, AND CALL THE APPROPRIATE TOOLS (`done`, `need_input`, or `restart_research`) AS NEEDED TO PROVIDE CLEAR AND ACCURATE RESPONSES. ALWAYS PROVIDE RESPONSES THAT ARE GENERALIZED AND DO NOT MENTION THE COMPANY NAME DIRECTLY.
    """  # noqa: E501
    )

    few_shot_examples = []

    def __init__(
        self,
        state: AgentsState,
        model: LLMModels = LLMModels.GPT_4O,
        agent_framework: AgentFrameWork = AgentFrameWork.PROMPT,
        cli_print: bool = False,
    ) -> None:
        super().__init__(state=state, model=model, agent_framework=agent_framework, cli_print=cli_print)
        self.logger = get_logger(__name__, stream_output=sys.stdout)
        self.research_results = []

    def restart_research(self) -> None:
        """Tool that indicates the restart of the research process."""
        self.logger.info("Restart research tool has been chosen.")

        self.research_results = []

        return "Research process restarted."

    def query_ESG_document(self, query: str) -> str:
        """Tool to query ESG document using llama-index query engine."""
        self.logger.info(f"Query walmart ESG document tool has been chosen with question: {query}")

        try:
            query_result = ESGQueryEngine(model=self.model).query(query)
        except Exception as e:
            self.logger.info(str(e))

        self.logger.info(f"Query was succesful with result: {query_result}")

        self.research_results.append(f"{query}: {query_result}")

        return f"""
            Research result for query: {query}

            {query_result}
        """

    def need_input(self, question: str) -> tuple[str, str]:
        """Tool to request input from the user."""
        self.logger.info("Needs input tool has been called")
        return (str(self.RESEARCH_NEED_INPUT), question)

    def done(self, message) -> tuple[str, str]:
        """Tool to indicate that the research agent has finished its task."""
        self.logger.info("Done tool has been called")

        self.state.research_results.extend(self.research_results)

        return (str(self.RESEARCH_DONE), message)

    @property
    def system_prompt(self) -> str:
        return self.prompt_template.format(agents_state=self.state.get_state_string())

    @property
    def tools(self) -> list[FunctionTool]:
        return [
            FunctionTool.from_defaults(fn=self.need_input, name="need_input", return_direct=True),
            FunctionTool.from_defaults(fn=self.query_ESG_document, name="query_ESG_document"),
            FunctionTool.from_defaults(fn=self.done, name="done", return_direct=True),
        ]

    @property
    def agent_description(self) -> str:
        return """The research agent can query information from the company’s ESG document to provide insights on
        sustainability efforts, social responsibility initiatives, and governance practices. It can extract key metrics,
        track compliance, and support decision-making by delivering relevant data from the document."""
