import copy
import sys

import pandas as pd
from llama_index.core import PromptTemplate
from llama_index.core.tools import FunctionTool

from agentic_copilot.models.utils.agent_base import AgentBase, AgentFrameWork
from agentic_copilot.models.utils.agents_util import (
    AgentsState,
    Speaker,
    get_logger,
)
from agentic_copilot.models.utils.llm_utils import LLMModels


class CalculationAgent(AgentBase):
    id: Speaker = Speaker.CALCULATION.value
    CALCULATION_DONE = "CALCULATION_DONE"
    CALCULATION_NEED_INPUT = "CALCULATION_NEED_INPUT"

    prompt_template = PromptTemplate(
        """
        YOU ARE THE **CALCULATOR AGENT** IN A MULTI-AGENTIC SYSTEM, A DATA ANALYSIS EXPERT RESPONSIBLE FOR PERFORMING CALCULATIONS ON PANDAS DATAFRAMES. YOUR MAIN TASK IS TO INTERPRET USER REQUESTS AND CARRY OUT PRECISE DATA ANALYSIS AND CALCULATIONS. ALL THE DATA REQUIRED FOR YOUR TASK HAS ALREADY BEEN QUERIED AND IS AVAILABLE IN THE DATAFRAMES YOU RECEIVE. YOU MUST USE THE AVAILABLE TOOLS TO PERFORM YOUR TASK EFFECTIVELY AND RETURN THE RESULT ACCURATELY.

        ### TOOLS AVAILABLE ###

        - **`secure_calculation`**: USE THIS TOOL TO EXECUTE PYTHON CODE SECURELY IN AN INDEPENDENT ENVIRONMENT, SPECIFICALLY FOR PANDAS OPERATIONS ON THE DATAFRAMES PROVIDED.
        - **`need_input`**: USE THIS TOOL IF YOU REQUIRE CLARIFICATION FROM THE USER, SUCH AS AMBIGUOUS INSTRUCTIONS OR MISSING INFORMATION.
        - **`done`**: ONCE YOU HAVE COMPLETED THE CALCULATION, USE THIS TOOL TO FINALIZE AND STORE THE RESULT, MAKING IT AVAILABLE TO THE ORCHESTRATOR AGENT.

        ### CHAIN OF THOUGHT TO EXECUTE CALCULATIONS ###

        1. **ANALYZE THE USER REQUEST**: THOROUGHLY READ THE USER'S QUERY AND UNDERSTAND WHICH SPECIFIC VALUE OR CALCULATION THEY NEED FROM THE PROVIDED DATAFRAME(S).
        - IF ANY PART OF THE REQUEST IS UNCLEAR, IMMEDIATELY USE THE `need_input` TOOL TO REQUEST CLARIFICATION.
        2. **CHECK DATAFRAME STRUCTURE**: REVIEW THE DATAFRAME(S) PROVIDED, INCLUDING COLUMN NAMES, DATA TYPES, AND ROWS, TO UNDERSTAND THE STRUCTURE AND IDENTIFY RELEVANT DATA FOR THE CALCULATION.
        3. **USE PANDAS FOR CALCULATIONS**:
        - CONSTRUCT THE APPROPRIATE PANDAS CODE USING THE `secure_calculation` TOOL TO EXECUTE THE CALCULATION BASED ON THE REQUEST.
        - IMPORT THE MODULES USED
        - ALWAYS EXECUTE THE CODE WITH ONE TOOL USAGE EVEN WHERE MULTIPLE VALUE WERE ASKED.
        - IF NECESSARY PUT IT IN A JSON LIKE VARIABLE AS KEY-VALUE PAIRS WITH LOGICAL KEY NAMES
        4. **VALIDATE RESULTS**:
        - VERIFY IF THE OUTPUT FROM THE CALCULATION MATCHES EXPECTATIONS AND THE USER'S REQUEST.
        - IF THE RESULT IS INCORRECT OR UNCLEAR, ADJUST THE CALCULATION AND RE-RUN IT.
        - IF UNCERTAINTY REMAINS, USE THE `need_input` TOOL TO ASK THE USER FOR MORE DETAILS OR CLARIFICATIONS.
        5. **FINALIZE THE RESULT**: USE THE `done` TOOL TO STORE THE FINAL CALCULATION RESULT AND SEND IT TO THE QUERY ORCHESTRATOR FOR FURTHER ACTION.

        ### WHAT NOT TO DO ###

        - **NEVER FORGET TO USE THE `done` TOOL**: ENSURE THAT YOU ALWAYS FINALIZE THE CALCULATION BY STORING THE RESULT USING THE `done` TOOL BEFORE RETURNING ANY OUTPUT.
        - **DO NOT EXECUTE INCOMPLETE OR INCORRECT CALCULATIONS**: ALWAYS DOUBLE-CHECK YOUR CODE AND OUTPUT BEFORE FINALIZING. IF UNCLEAR, REQUEST CLARIFICATION WITH THE `need_input` TOOL.
        - **DO NOT EXECUTE SEPARATE LINES OF CODE**: WHEN USING THE `secure_calcultion` TOOL MAKE SURE THE CODE YOU ARE EXECUTING IS NOT USING VARIABLES AND OTHER DEPENDENCIES FROM PREVIOUS CODE EXECUTIONS.
        - **AVOID ASSUMPTIONS**: IF THE REQUEST IS UNCLEAR, NEVER GUESS THE USER'S INTENT—ALWAYS ASK FOR MORE INFORMATION USING THE `need_input` TOOL.

        ### TEMPLATE FOR TASK EXECUTION ###

        #### CHAT HISTORY:
        {chat_history}

        #### DATAFRAME VARIABLE NAMES AND THEIR HEADS:
        {dataframe_heads}

        #### STEPS TO PERFORM:

        1. **ANALYSIS OF THE REQUEST**:
        - CAREFULLY UNDERSTAND THE TASK AND IDENTIFY THE RELEVANT COLUMNS AND OPERATIONS REQUIRED FROM THE DATAFRAME(S).
        - IF ANYTHING IS UNCLEAR, USE THE `need_input` TOOL TO REQUEST CLARIFICATION.

        2. **PANDAS CALCULATION**:
        - USE THE `secure_calculation` TOOL TO WRITE AND EXECUTE THE PANDAS CODE THAT SOLVES THE REQUESTED CALCULATION.
        - BE SURE TO REFER TO THE VARIABLES FROM THE `DATAFRAME VARIABLE NAMES AND HEADS` SECTION.
        - YOU CAN CALCULATE MULTIPLE VALUES IN ONE CODE AND THEN PUT IT IN A JSON LIKE STRING

        3. **VERIFY THE CALCULATION RESULT**:
        - REVIEW THE OUTPUT FROM THE `secure_calculation` TOOL TO ENSURE IT IS CORRECT AND SATISFIES THE USER'S REQUEST.
        - IF THE RESULT IS INCORRECT OR AMBIGUOUS, REVISE THE CODE AND RUN AGAIN. IF NEEDED, REQUEST CLARIFICATION USING `need_input`.

        4. **FINALIZE WITH `done`**:
        - ONCE THE RESULT IS VERIFIED, USE THE `done` TOOL TO RETURN THE FINAL RESULT TO THE QUERY ORCHESTRATOR AGENT.

        ### EXAMPLE SCENARIOS AND CODE ###

        #### EXAMPLE 1:
        **User Request**: "What is difference of the means for the Water Usage and Anti Bribery compliance?"

        1. **ANALYZE REQUEST**: The user asks for the average of the `price` column.
        2. **CALCULATE USING PANDAS**:
        ```python
            command="result = water_usage_sikkim_india_106['value'].mean() - anti_bribery_compliance_minnesota_united_states_31['value'].mean()"

    """  # noqa: E501
    )

    def __init__(
        self,
        state: AgentsState,
        model: LLMModels = LLMModels.CLAUDE_3_5_HAIKU,
        agent_framework: AgentFrameWork = AgentFrameWork.PROMPT,
        cli_print: bool = False,
    ) -> None:
        super().__init__(state=state, model=model, agent_framework=agent_framework, cli_print=cli_print)
        self.logger = get_logger(__name__, stream_output=sys.stdout)
        self.calculation_result: pd.DataFrame = None

    def need_input(self, question_to_user) -> tuple[str, str]:
        """A tool to ask question from user, if you are unsure how to calculate the value they requested."""
        self.logger.info(f"Need input tool has been used with question: {question_to_user}")

        return self.CALCULATION_NEED_INPUT, question_to_user

    def secure_calculation(self, command: str) -> str:
        """Use this tool to do pandas operations on DataFrames queried prior.
        DEFINE THE COMMAND AS A PYTHON CODE SNIPPET USING THE QUERIED PADNAS DATAFRAMES.
        IF MORE DATA IS REQUESTED RETURN A DICT OBJECT LIKE:
            {'max': 23, 'mean': 12, 'variance': 3}
        DOUBLE CHECK THE VARIABLE NAMES BEFORE EXECUTING A COMMAND.
        DEFINE THE QUERIES LIKE THIS, STORE THE RESULT IN A VARIABLE CALLED 'result':
            command="result = water_usage_sikkim_india_106['value'].mean() - anti_bribery_compliance_minnesota_united_states_31['value'].mean()"
        """  # noqa: E501
        self.logger.info(f"Pandas engine tool was used with command: {command}")
        dataframes = self.state.queried_data

        code_exec_vars = copy.deepcopy(dataframes)

        try:
            exec(command, code_exec_vars)
            self.calculation_result = code_exec_vars["result"]
            self.logger.info(f"Result of the calculation is: {self.calculation_result}")

        except Exception as e:
            self.logger.info(f"Calculation threw an exception: {str(e)}")
            return f"""
                Some error occured while trying to execute command: {command}
                Error: {str(e)}
            """

        return f"""
            Calculation done
            Result:
                {self.calculation_result}
            Check the result of he calculation and call the done tool if you think its correct or try to use this tool again with a different command.
        """  # noqa: E501

    def done(self) -> str:
        """Use this tool when you are finished with the calculation."""
        self.logger.info("Calculation done tool has been chosen")

        self.state.calculation_results.append(str(self.calculation_result))

        return (self.CALCULATION_DONE, "")

    @property
    def system_prompt(self) -> str:
        dataframe_heads = "\n\n".join(f"{df[0]}:\n{df[1].head(2)}" for df in self.state.queried_data.items())
        self.logger.info(f"Dataframe heads: {dataframe_heads}")
        return self.prompt_template.format(chat_history=self.state.chat_history, dataframe_heads=dataframe_heads)

    @property
    def tools(self) -> list[FunctionTool]:
        return [
            FunctionTool.from_defaults(fn=self.need_input, name="need_input", return_direct=True),
            FunctionTool.from_defaults(fn=self.secure_calculation, name="secure_calculation"),
            FunctionTool.from_defaults(fn=self.done, name="done", return_direct=True),
        ]

    @property
    def agent_description(self) -> str:
        return """This agent processes data retrieved by Query agents, stored in a dictionary of type
        dict[str, DataFrame]. It uses pandas functions and its toolset to compute results efficiently."""
