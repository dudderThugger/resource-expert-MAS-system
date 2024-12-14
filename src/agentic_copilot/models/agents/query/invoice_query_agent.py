import sys
from pathlib import Path

import pandas as pd
from llama_index.core import PromptTemplate
from llama_index.core.tools import FunctionTool

from agentic_copilot.models.utils.agent_base import AgentFrameWork, QueryAgentBase
from agentic_copilot.models.utils.agents_util import (
    AgentsState,
    Speaker,
    get_logger,
)
from agentic_copilot.models.utils.llm_utils import LLMModels


class InvoiceQueryAgent(QueryAgentBase):
    id: str = Speaker.INVOICE_QUERY
    INVOICE_QUERY_NEED_INPUT = "INVOICE_QUERY_NEED_INPUT"
    INVOICE_QUERY_DONE = "INVOICE_QUERY_DONE"

    def __init__(
        self,
        state: AgentsState,
        model: LLMModels = LLMModels.CLAUDE_3_5_HAIKU,
        agent_framework: AgentFrameWork = AgentFrameWork.PROMPT,
        cli_print: bool = False,
    ):
        super().__init__(state=state, model=model, agent_framework=agent_framework, cli_print=cli_print)
        self._create_df()
        self.queried_data = {}
        self.different_values = self._get_attribs_with_values()
        self.records_str = (
            self.df.sample(40).drop_duplicates(subset=["status"]).drop_duplicates(subset=["country", "submitted_by"])
        )
        self.logger = get_logger(__name__, stream_output=sys.stdout)

    agent_prompt_template = PromptTemplate(
        """
        YOU ARE A DATA QUERY AGENT IN A MULTI-AGENT SYSTEM, TASKED WITH RETRIEVING AND FILTERING DATA FROM CSV FILES USING PANDAS FUNCTIONS. YOUR GOAL IS TO FORMULATE PANDAS QUERIES BASED ON THE INSTRUCTIONS PROVIDED AND RETURN THE RESULTING DATAFRAME. YOU HAVE TOOLS TO HELP IDENTIFY DISTINCT VALUES IN COLUMNS IF NEEDED.

        ### KEY RULES AND RESTRICTIONS ###

        - **DATA ACCESS**: YOU CAN ONLY ACCESS DATA THROUGH THE `pandas_engine` TOOL, WHICH ALLOWS YOU TO EXECUTE DATA QUERIES IN PANDAS.
        - **TOOLS AVAILABLE**:
        - **`pandas_engine`**: USE THIS TOOL TO EXECUTE THE PANDAS QUERY BASED ON GIVEN PARAMETERS AND STORES IT IN A VARIABLE ON NAME 'valriable_name' PARAMETER FOR OTHER AGENTS TO ACCESS.
        - **`need_input`**: USE THIS TOOL TO REQUEST ADDITIONAL CLARIFICATION FROM THE USER IF THE QUERY REQUIREMENTS ARE UNCLEAR.
        - **`done`**: USE THIS TOOL TO FINALIZE YOUR ACTION AND RETURN THE RESULTING DATAFRAME.


        ### CHAIN OF THOUGHT TO FORMULATE AND EXECUTE QUERIES ###

        1. **UNDERSTAND THE QUERY**: COMPREHEND THE USER'S INSTRUCTION AND IDENTIFY THE SPECIFIC CONDITIONS TO QUERY, INCLUDING `status`, `service_month`, AND `site_name`. USE `need_input` IF THE QUERY IS UNCLEAR.
        2. **ASK FOR CLARIFICATIONS** (OPTIONAL): IF YOU NEED CLARIFICATION RELATED TO ONE OF THE QUERIES PARAMETER ASK THE USER WITH THE `need_input` TOOL, ALWAYS GIVE THEM VALID OPTIONS TO CHOOSE FROM.
        3A. **FORMULATE THE PANDAS QUERY**: ONCE YOU HAVE THE REQUIRED INFORMATION, USE `pandas_engine` TO CONSTRUCT THE QUERY IN PANDAS SYNTAX. DO NOT STORE THE RESULT IN A VARIABLE.
            EXAMPLES:
            - `df[(df['country'] == 'Brazil') & (df['status'] == 'SUBMITTED') & (df['service_month'] == 'APR-2022')]`
            - `df[(df['site_name'] == 'Western Australia_Australia_83') & (df['status'] == 'POSTED') & (df['service_month'].str.contains('2022'))]`
        3B. **NAME THE QUERY**: GIVE A DESCRIPTIVE NAME THAT INDICATES THE QUERY'S PURPOSE. FOR EXAMPLE:
        - `variable_name="western_australia_australia_83_adam_s_posted_2022"`, query="df[(df['site_name'] == 'Western Australia_Australia_83') & (df['status'] == 'POSTED') & (df['submitted_by'] == 'Adam S.') & (df['service_month'].str.contains('2022'))]"`
        4. **VALIDATE RESULTS**: AFTER EXECUTING THE QUERY VALIDATE THE RESULTS TO MAKE SURE EVERY VALUE IS AS EXPECTED.
        5. **FINALIZE THE QUERY**: AFTER THE VALIDATION OF THE QUERY RESULTS, USE `done` TOOL TO SIGNAL COMPLETION AND RETURN THE DATAFRAME.

        ### WHAT NOT TO DO ###

        - **DO NOT ACCESS DATA WITHOUT USING `pandas_engine`**.
        - **DO NOT STORE QUERY RESULTS IN VARIABLES**; WHEN FORMULATING A QUERY CODE FOR THE `pandas_engine` TOOL DON'T STORE VALUES IN VARIABLES SIMPLY RETURN THEM AS SHOWN IN THE EXAMPLES.
        - **AVOID UNNECESSARY CLARIFICATION REQUESTS**; DON'T TRY TO CLARIFY EVERYTHING, BUT IF YOU USE THE `need_input` TOOL GIVE THE USER OPTIONS TO CHOOSE FROM THAT ARE VALID VALUES.

        ### FEW-SHOT-EXAMPLES ###

        {few_shot_examples}

        ### DIFFERENT VALUES OF SOME COLUMNS ###

        {different_values}

        ### EXAMPLE RECORDS ###

        {records}

        ### SUMMARY ###

        FORMULATE PANDAS QUERIES USING THE TOOLS PROVIDED. RETURN RESULTS DIRECTLY WITHOUT STORING THEM IN VARIABLES IN THE PANDAS QUERIES.
    """  # noqa: E501
    )

    few_shot_examples = ",\n".join(
        [
            """
            ### Example 1:

            #### Instruction:
            Retrieve records from site `Ontario_Canada_202` for services with status `POSTED` during the month of `MAR-2023`.

            #### Step-by-step Solution:

            1. **UNDERSTAND THE QUERY**
            - Identify the specific parameters for the query:
                - **`status`** is specified as "POSTED".
                - **`service_month`** is specified as "MAR-2023".
                - **`site_name`** is specified as "Ontario_Canada_202".
            - The instruction is clear with no ambiguities, so `need_input` is not required.

            2. **ASK FOR CLARIFICATIONS** *(Optional, Not Needed Here)*
            - All required parameters are present, so no clarifications are needed.

            3A. **FORMULATE THE PANDAS QUERY**
            - Construct the query using `pandas_engine` to filter `df` by the specified conditions:
                ```python
                df[(df['status'] == 'POSTED') & (df['service_month'] == 'MAR-2023') & (df['site_name'] == 'Ontario_Canada_202')]
                ```

            3B. **NAME THE QUERY**
            - Choose a descriptive name to represent the purpose of the query:
                ```python
                variable_name = "ontario_canada_202_completed_mar_2023"
                query = "df[(df['status'] == 'POSTED') & (df['service_month'] == 'MAR-2023') & (df['site_name'] == 'Ontario_Canada_202')]"
                ```

            4. **VALIDATE RESULTS**
            - Execute the query and inspect the resulting DataFrame to validate:
                - Ensure all records have a `status` of "POSTED".
                - Confirm the `service_month` is "MAR-2023" for each entry.
                - Check that each record's `site_name` is "Ontario_Canada_202".
            - If any discrepancies are found, adjust the query and re-validate.

            5. **FINALIZE THE QUERY**
            - Once the results are validated and accurate, use the `done` tool to signal completion and return the filtered DataFrame for further analysis or display.
        """  # noqa: E501
        ]
    )

    def _create_df(self) -> None:
        df_full = pd.read_csv(Path("data/synth_invoice_data_v2.csv"), encoding="ISO-8859-1")
        self.df = df_full.astype(
            {
                "site_name": "string",
                "state": "string",
                "country": "string",
                "service_month": "string",
                "invoice_name": "string",
                "submitted_by": "string",
                "status": "string",
            }
        )

    def _get_attribs_with_values(self) -> str:
        return "\n".join(
            [
                f"\t{attrib} " + ", ".join([value for value in self.df[attrib].drop_duplicates()])
                for attrib in self.df.columns
                if attrib not in ["Unnamed: 0", "invoice_name", "state"]
            ]
        )

    def need_input(self, question: str) -> tuple[str, str]:
        """Use this tool when you can't answer the query for instance you can't find the requested invoice.
        The question parameter must contain a precise question you want to ask from the user.
        Give them options in the question parameter to choose from make sure the values are existent.
        """
        self.logger.info(f"Need input tool has been chosen with question: {question}")

        return str(self.INVOICE_QUERY_NEED_INPUT), question

    def pandas_engine(self, variable_name: str, query: str) -> str:
        """Use this tool to do pandas operations, but make sure that the parameters are existing.
        DEFINE THE QUERY AS A PYTHON CODE SNIPPET USING PANDAS DATAFRAME 'df'!!
        THE FIRST PARAMETER IS THE NAME OF THE VARIABLE WE WILL STORE THE RESULT OF THE QUERY INTO, MAKE SURE IT HAS A
        TELLING NAME, WHICH REFLECTS TO WHAT THE ORIGINAL QUERY WAS. DEFINE THE QUERIES LIKE THIS, DONT STORE THE
        RESULT IN A VARIABLE JUST RETURN WITH IT:
            variable_name="santa_catarina_2024_posted", query="df[(df['state'] == 'Santa Catarina') & (df['service_month'].str.contains('2024')) & (df['status'] == 'POSTED')]"
        """  # noqa: E501
        self.logger.info(f"Pandas engine tool was used with query: {query}")

        try:
            local_vars = {"df": self.df}
            result = eval(query, globals(), local_vars)
        except Exception as e:
            self.logger.info(f"Query threw an exception: {str(e)}")
            return f"""
                Some error occured while trying to execute query: {query}
                Error: {str(e)}
            """

        self.queried_data[variable_name] = result

        self.logger.info(f"Result of the query is: {self.queried_data}")

        return f"""
            Query was succesful!
            Result:
                {self.queried_data[variable_name]}
            Now validate the reults then call query_done tool to proceed with the execution
        """

    def done(self) -> str:
        """Use this tool when you are finished with the query of the requested data and the results are verified."""
        self.logger.info("Invoice query done tool has been chosen")

        self.state.queried_data.update(self.queried_data)

        return (str(self.INVOICE_QUERY_DONE), "")

    @property
    def query_description(self) -> str:
        return f"""
        {self.agent_description}

        Attributes of an invoice record:
        {self.different_values}

        Some sample invoice records:
        {self.records_str}
        """

    @property
    def agent_description(self) -> str:
        return """
        This agent queries invoice records, including their status (POSTED, IN-PROCESS, PROCESSED, SUBMITTED),
        unique invoice names, and associated site details (country, state). Invoices are issued monthly at a site
        and submitted by an employee.
        """

    @property
    def tools(self) -> list[FunctionTool]:
        return [
            FunctionTool.from_defaults(fn=self.need_input, name="need_input", return_direct=True),
            FunctionTool.from_defaults(fn=self.pandas_engine, name="pandas_engine"),
            FunctionTool.from_defaults(fn=self.done, name="done", return_direct=True),
        ]

    @property
    def system_prompt(self) -> str:
        return self.agent_prompt_template.format(
            few_shot_examples=self.few_shot_examples,
            different_values=self.different_values,
            records=self.records_str,
        )
