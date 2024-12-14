import sys
from pathlib import Path

import pandas as pd
from llama_index.core import PromptTemplate
from llama_index.core.tools import FunctionTool

from agentic_copilot.models.agents.query.client_datastream_matching_engine import (
    ClientDataStreamMatchingEngine,
)
from agentic_copilot.models.utils.agent_base import AgentFrameWork, QueryAgentBase
from agentic_copilot.models.utils.agents_util import (
    AgentsState,
    Speaker,
    get_logger,
)
from agentic_copilot.models.utils.llm_utils import LLMModels


class DataStreamQueryAgent(QueryAgentBase):
    id: str = Speaker.DATASTREAM_QUERY
    DS_AGENT_NEED_INPUT = "DS_AGENT_NEED_INPUT"
    DS_QUERY_DONE = "DS_QUERY_DONE"

    agent_prompt_template = PromptTemplate(
        """
        YOU ARE A DATA QUERY AGENT IN A MULTI-AGENT SYSTEM, TASKED WITH RETRIEVING AND FILTERING DATA FROM CSV FILES USING PANDAS FUNCTIONS. YOUR GOAL IS TO FORMULATE PANDAS QUERIES BASED ON THE INSTRUCTIONS PROVIDED AND RETURN THE RESULTING DATAFRAME. YOU HAVE TOOLS TO HELP IDENTIFY DISTINCT VALUES IN COLUMNS IF NEEDED.

        ### KEY RULES AND RESTRICTIONS ###

        - **DATA ACCESS**: YOU CAN ONLY ACCESS DATA THROUGH THE `pandas_engine` TOOL, WHICH ALLOWS YOU TO EXECUTE DATA QUERIES IN PANDAS.
        - **COLUMN VALUE CHECK**: USE THE `get_dif_values_of_column` OR `find_datastreams` TOOLS IF YOU NEED TO IDENTIFY OR CONFIRM VALUES IN SPECIFIC COLUMNS. THESE TOOLS ARE OPTIONAL.
        - **SINGLE QUERIES**: ALWAYS SOLVE ONE INSTRUCTION IN A SINGLE QUERY
        - **TOOLS AVAILABLE**:
        - **`pandas_engine`**: USE THIS TOOL TO EXECUTE THE PANDAS QUERY BASED ON GIVEN PARAMETERS AND STORES IT IN A VARIABLE ON NAME 'valriable_name' PARAMETER FOR OTHER AGENTS TO ACCESS.
        - **`get_dif_values_of_column`**: USE THIS TOOL TO LIST ALL UNIQUE VALUES IN A SPECIFIC COLUMN (e.g., `data_stream`, `site_name`, etc.) WHEN YOU NEED TO KNOW WHAT VALUES EXIST.
        - **`find_datastreams`**: USE THIS TOOL TO GET DATASTREAMS THAT ARE CLOSEST TO A STRING OR TO VALIDATE DATASTREAM.
        - **`need_input`**: USE THIS TOOL TO REQUEST ADDITIONAL CLARIFICATION FROM THE USER IF THE QUERY REQUIREMENTS ARE UNCLEAR.
        - **`done`**: USE THIS TOOL TO FINALIZE YOUR ACTION AND RETURN THE RESULTING DATAFRAME.

        ### CHAIN OF THOUGHT TO FORMULATE AND EXECUTE QUERIES ###

        1. **UNDERSTAND THE QUERY**: COMPREHEND THE USER'S INSTRUCTION AND IDENTIFY THE SPECIFIC CONDITIONS TO QUERY, INCLUDING `data_stream`, `service_month`, `type`, AND `site_name`. USE `need_input` IF THE QUERY IS UNCLEAR.
        2. **COLUMN VALUE CHECK**: USE `get_dif_values_of_column` TO FETCH UNIQUE VALUES OF COLUMN. USE `find_datastreams` TO FIND DATASTREAMS SIMILAR TO THE ONE THE USER ASKED.
        3. **ASK FOR CLARIFICATIONS** (OPTIONAL): IF YOU NEED CLARIFICATION RELATED TO ONE OF THE QUERIES PARAMETER ASK THE USER WITH THE `need_input` TOOL, ALWAYS GIVE THEM OPTIONS TO CHOOSE FROM.
            USE `get_dif_values_of_column` and `find_datastreams` TOOLS TO GET POSSIBLE OPTIONS, DON'T COME UP WITH YOURSELF.
        4A. **FORMULATE THE PANDAS QUERY**: ONCE YOU HAVE THE REQUIRED INFORMATION, USE `pandas_engine` TO CONSTRUCT THE QUERY IN PANDAS SYNTAX. DO NOT STORE THE RESULT IN A VARIABLE.
            ALWAYS ASSUME THE USER MEANT ACTUAL RECORDS IF THEY DIDN'T DISCLOSE FORECASTED SPECIFICALLY!
            EXAMPLES:
            - `df[(df['type'] == 'Actual') & (df['data_stream'] == 'Electricity Cost') & (df['service_month'] == 'APR-2022')]`
            - `df[(df['type'] == 'Forecasted') & (df['data_stream'] == 'Compostable Waste') & (df['service_month'].str.contains('2022')) & (df['site_name'] == 'Illinois_United States_122')]`
        4B. **NAME THE QUERY**: GIVE A DESCRIPTIVE NAME THAT INDICATES THE QUERY'S PURPOSE. FOR EXAMPLE:
        - `variable_name="water_usage_jan_feb_2024"`, query="df[(df['type'] == 'Actual') & (df['data_stream'] == 'Water Usage') & ((df['service_month'] == 'JAN-2024') | (df['service_month'] == 'FEB-2024'))]"`
        5. **VALIDATE RESULTS**: AFTER EXECUTING THE QUERY VALIDATE THE RESULTS TO MAKE SURE EVERY VALUE IS AS EXPECTED.
        6. **FINALIZE THE QUERY**: AFTER THE VALIDATION OF THE QUERY RESULTS, USE `done` TOOL TO SIGNAL COMPLETION AND RETURN THE DATAFRAME.

        ### WHAT NOT TO DO ###

        - **DO NOT ACCESS DATA WITHOUT USING `pandas_engine`**.
        - **DO NOT STORE QUERY RESULTS IN VARIABLES**; WHEN FORMULATING A QUERY CODE FOR THE `pandas_engine` TOOL DON'T STORE VALUES IN VARIABLES SIMPLY RETURN THEM AS SHOWN IN THE EXAMPLES.
        - **AVOID UNNECESSARY CLARIFICATION REQUESTS**; DON'T TRY TO CLARIFY EVERYTHING, BUT IF YOU USE THE `need_input` TOOL GIVE THE USER OPTIONS TO CHOOSE FROM THAT ARE VALID VALUES.
        - **MULTIPLE QURIES**; ALWAYS SOLVE THE INSTRUCTIONS IN A SINGLE QUERY

        ### FEW-SHOT-EXAMPLES ###

        {few_shot_examples}

        ### ATTRIBUTES OF A RECORD ###

        {attributes}

        ### EXAMPLE RECORDS ###

        {records}

        ### SUMMARY ###

        FORMULATE PANDAS QUERIES USING THE TOOLS PROVIDED, AND USE `get_dif_values_of_column` OR `find_datastreams` WHEN YOU NEED TO IDENTIFY OR CONFIRM COLUMN VALUES. RETURN RESULTS DIRECTLY WITHOUT STORING THEM IN VARIABLES IN THE PANDAS QUERIES.
    """  # noqa: E501
    )

    few_shot_examples = ",\n".join(
        [
            """
            ## EXMAMPLE 1:
            ### Instruction:
            Fetch electricity records measured at site `Nunavut_Canada_118` in 2022.

            ### Step-by-step Solution:
            1. **Use the `find_datastreams` tool** to search for the 'electricity' datastream.
            Command: `find_datastreams(datastream='electricity')`

            2. **Tool returns multiple matches** related to 'electricity'.
            Example: `['Renewable Energy Usage', 'Electric Vehicle Charging Stations', 'Employee Commute Emissions']`

            3. **Use the `need_input` tool** to ask the user which specific datastream they meant.
            Command: `need_input(question="I found multiple datastreams for 'electricity': 'Renewable Energy Usage', 'Electric Vehicle Charging Stations', 'Employee Commute Emissions'. Which one do you mean?")`
            """,  # noqa: E501
            """
            ## EXMAMPLE 2:
            ### Instruction:
            Fetch Renewable Energy Usage records measured at site `Nunavut_Canada_118` in 2022.

            ### Step-by-step Solution:
            1. **Use the `find_datastreams` tool** to search for the 'Renewable Energy Usage' datastream.
            Command: `find_datastreams(datastream='Renewable Energy Usage')`

            2. **Tool returns with a single match** related to 'Renewable Energy Usage'.
            Example: `['Renewable Energy Usage']`

            3. **Use the pandas pandas_engine tool** to query for the specified records:
            Command: `pandas_engine(variable_name='renewable_energy_usage_nunavut_canada_118_2022', query="df[(df[type] = 'Actual') & (df['data_stream'] == 'Renewable Energy Usage') & (df['service_month'].str.contains('2022')) & (df['site_name'] == 'Nunavut_Canada_118')]")`

            4 **Tool returns with a string containing the queried data's head**
            Example: `Query was succesful!
            Result:
                               client_id    data_stream             site_name
                23142          4            Renewable Energy Usage  Rio Grande do Sul_Brazil_207
                23143          4            Renewable Energy Usage  Rio Grande do Sul_Brazil_207
                23262          4            Renewable Energy Usage  Rio Grande do Sul_Brazil_207

                       state              country     service_month     type        value
                23142  Rio Grande do Sul  Brazil      APR-2022          Actual      40717.998665
                23143  Rio Grande do Sul  Brazil      MAY-2022          Actual      46291.999575
                23262  Rio Grande do Sul  Brazil      JUN-2022          Actual      19744.399717  '
            Now call query_done tool to proceed with the execution`

            5. **Use the `done` tool** to indicate that the query was successful.
            Command: `done() -> {str(DS_QUERY_DONE)}`
            """,  # noqa: E501
        ]
    )

    def __init__(
        self,
        state: AgentsState,
        model: LLMModels = LLMModels.CLAUDE_3_5_HAIKU,
        agent_framework: AgentFrameWork = AgentFrameWork.PROMPT,
        cli_print: bool = False,
    ) -> None:
        super().__init__(state=state, model=model, agent_framework=agent_framework, cli_print=cli_print)
        self._create_df()
        self.queried_data = {}
        self.attributes_str = ", ".join(self.df.columns)
        self.records_str = self.df.sample(10)
        self.logger = get_logger(__name__, stream_output=sys.stdout)

    def _create_df(self) -> None:
        df_full = pd.read_csv(Path("data/datastreams_full_synth.csv"), encoding="ISO-8859-1")
        df_full_typed = df_full.astype(
            {
                "client_id": "int64",
                "data_stream": "string",
                "site_name": "string",
                "state": "string",
                "country": "string",
                "service_month": "string",
                "type": "string",
            }
        )
        self.df = df_full_typed[df_full_typed["client_id"] == int(self.state.user_id)]

    def need_input(self, question: str) -> tuple[str, str]:
        """Use this tool when you can't answer the query for instance you can't resolve datastream value.
        The question parameter must contain the precise question you want to ask from the user.
        Give them options to choose from make sure the values are existent.
        """
        self.logger.info(f"Need input tool has been chosen with question: {question}")

        return str(self.DS_AGENT_NEED_INPUT), question

    def find_datastreams(self, datastream: str) -> list[str]:
        """This tool return the most similar datastreams to the 'datastream' parameter"""

        self.logger.info(
            f"Find datastream tool has been chosen for datastream: {datastream} and user {self.state.user_id}"
        )

        datastream_matching_engine = ClientDataStreamMatchingEngine(self.state.user_id)
        matched_datastreams = datastream_matching_engine.match_datastream(datastream)

        self.logger.info(f"{matched_datastreams}")

        return matched_datastreams

    def get_dif_values_of_column(self, column: str) -> pd.Series:
        """Use this tool to get the existing values of a column so you can choose from these values for the query"""
        self.logger.info(f"Get different values of column was used for column: {column}")

        values = self.df[[column]].drop_duplicates()

        self.logger.info(f"The different values of the column are: {values}")

        return values

    def pandas_engine(self, variable_name: str, query: str) -> str:
        """Use this tool to do pandas operations, but make sure that the parameters are existing.
        DEFINE THE QUERY AS A PYTHON CODE SNIPPET USING PANDAS DATAFRAME 'df'!!
        THE FIRST PARAMETER IS THE NAME OF THE VARIABLE WE WILL STORE THE RESULT OF THE QUERY INTO, MAKE SURE IT HAS A TELLING NAME, WHICH REFLECTS TO WHAT THE ORIGINAL QUERY WAS.
        DEFINE THE QUERIES LIKE THIS, DONT STORE THE RESULT IN A VARIABLE JUST RETURN WITH IT:
            variable_name="eletricity_cost_april", query="df[(df['type'] == 'Actual') & (df['data_stream'] == 'Electricity Cost') & (df['service_month'].str.contains('APR'))]"
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
            Now validate the results then call the 'done' tool to finalize the query
        """

    def done(self) -> str:
        """Use this tool when you are done with the query and found records it was looking for.
        This tool will format the queried results and return it to the user.
        You must use this tool instead of returning an answer."""
        self.logger.info("Query done tool has been chosen")

        self.state.queried_data.update(self.queried_data)

        return (str(self.DS_QUERY_DONE), "")

    @property
    def query_description(self) -> str:
        return f"""
            {self.agent_description}

            Attributes of a datasream record:
            {self.attributes_str}

            Some sample datastream records:
            {self.records_str}
        """  # noqa: E501

    @property
    def agent_description(self) -> str:
        return """
            The Datastream Query Agent processes and analyzes user datastreams (HR, Energy, Emissions, etc.), each
            tied to a service month and value. It supports querying, filtering, and aggregating data for insights
            like trends, anomalies, and performance metrics.
        """

    @property
    def system_prompt(self) -> str:
        return self.agent_prompt_template.format(
            few_shot_examples=self.few_shot_examples,
            attributes=self.attributes_str,
            records=self.records_str,
        )

    @property
    def tools(self) -> list[FunctionTool]:
        return [
            FunctionTool.from_defaults(fn=self.need_input, name="need_input", return_direct=True),
            FunctionTool.from_defaults(fn=self.find_datastreams, name="find_datastreams"),
            FunctionTool.from_defaults(fn=self.get_dif_values_of_column, name="get_dif_values_of_column"),
            FunctionTool.from_defaults(fn=self.pandas_engine, name="pandas_engine"),
            FunctionTool.from_defaults(fn=self.done, name="done", return_direct=True),
        ]
