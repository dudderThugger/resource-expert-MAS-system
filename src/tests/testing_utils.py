import asyncio
import unittest
from pathlib import Path
from typing import Any, Callable, Coroutine

import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from openai.resources.chat.completions import ChatCompletion
from llama_index.core.llms import ChatMessage, MessageRole
from rapidfuzz import process

from agentic_copilot.models.utils.agents_util import Speaker
from agentic_copilot.models.utils.llm_utils import LLMModels, llm_factory_function

MAX_RETRIES = 5


class TestValues:
    def __init__(self) -> None:
        user_id = 0
        df = pd.read_csv(Path("data/datastreams_full_synth.csv"), encoding="ISO-8859-1")

        self.product_returns_brazil_jan_2023 = df[
            (df["client_id"] == user_id)
            & (df["data_stream"] == "Product Returns")
            & (df["country"] == "Brazil")
            & (df["service_month"] == "JAN-2023")
        ]

        self.forecasted_product_metrics_2022_2023 = df[
            (df["client_id"] == user_id)
            & (df["type"] == "Forecasted")
            & (df["data_stream"].isin(["Product Returns", "Product Defect Rate"]))
            & (df["service_month"].str.contains("2022") | df["service_month"].str.contains("2023"))
        ]

        self.query_result_brazil = df[
            (df["client_id"] == user_id)
            & (df["data_stream"] == "Cloud Storage Usage")
            & (df["type"] == "Actual")
            & (df["site_name"] == "Acre_Brazil_182")
        ]

        self.query_result_india = df[
            (df["client_id"] == user_id)
            & (df["data_stream"] == "Conflict Mineral Usage")
            & (df["type"] == "Actual")
            & (df["site_name"] == "Assam_India_150")
        ]

        self.northern_territory = df[
            (df["data_stream"] == "Production Line 3 - Units Produced")
            & (df["service_month"].str.contains("2022"))
            & (df["type"] == "Actual")
            & (df["site_name"] == "Northern Territory_Australia_32")
        ]

        self.alberta_canada = df[
            (df["data_stream"] == "Product Life Cycle Emissions")
            & (df["service_month"].str.contains("2022"))
            & (df["type"] == "Actual")
            & (df["site_name"] == "Alberta_Canada_268")
        ]

        self.query_plan_successful = [
            (
                Speaker.QUERY_ORCHESTRATOR.value,
                "Query the actual Cloud Storage Usage data for the Acre_Brazil_182 site",
            ),
            (
                Speaker.QUERY_ORCHESTRATOR.value,
                "Query the actual Conflict Mineral Usage data for the Assam_India_150 site",
            ),
            (
                Speaker.CALCULATION.value,
                "Perform a regression analysis with the Acre_Brazil_182 site's data as the independent variable and the Assam_India_150 site's data as the dependent variable",  # noqa: E501
            ),
        ]

        self.query_result_cost_of_goods = df[
            (df["client_id"] == user_id)
            & (df["data_stream"] == "Cost of Goods Sold")
            & (df["type"] == "Actual")
            & (df["site_name"] == "Tennessee_United States_249")
        ]

        self.plan_need_input = [
            (
                Speaker.QUERY_ORCHESTRATOR.value,
                "Query the actual cost data for the Tennessee_United States_249 site",
            )
        ]

        self.chat_history_need_input = [
            "Orchestrator agent to Planning agent: Query the actual cost data for the Tennessee_United States_249 site",
            "Planning agent to Orchestration agent: ('PLAN_DONE', [('query_orchestrator_agent', 'Query the actual cost data for the Tennessee_United States_249 site')])",  # noqa: E501
            "Orchestrator agent to QueryOrchestrator agent: Query the actual cost data for the Tennessee_United States_249 site",  # noqa: E501
            "QueryOrchestrator agent to DataStreamQueryAgent: Query the actual cost data for the Tennessee_United States_249 site",  # noqa: E501
            "QueryOrchestrator agent to Orchestrator agent: ('QUERY NEED INPUT', \"I found multiple datastreams related to 'cost': 'IT Energy Consumption', 'Maintenance Costs', 'Cost of Goods Sold', 'Fleet Fuel Consumption'. Which one do you mean?\")",  # noqa: E501
        ]


def get_similarity(sentence_x: str, sentence_y: str) -> float:
    x_list = word_tokenize(sentence_x)
    y_list = word_tokenize(sentence_y)

    sw = stopwords.words("english")
    l1 = []
    l2 = []

    x_set = {w for w in x_list if w not in sw}
    y_set = {w for w in y_list if w not in sw}

    rvector = x_set.union(y_set)

    for w in rvector:
        if w in x_set:
            l1.append(1)
        else:
            l1.append(0)
        if w in y_set:
            l2.append(1)
        else:
            l2.append(0)
    c = 0

    # cosine formula
    for i in range(len(rvector)):
        c += l1[i] * l2[i]
    return c / float((sum(l1) * sum(l2)) ** 0.5)


def check_similar_meaning(sentence_x: str, sentence_y: str) -> bool:
    similarity = get_similarity(sentence_x=sentence_x, sentence_y=sentence_y)
    print(f"Similarity is {similarity}")
    return similarity > 0.3


async def run_test_on_models(
    test_class: unittest.IsolatedAsyncioTestCase,
    test_case: Callable[[unittest.IsolatedAsyncioTestCase, str, str], list],
):
    test_models = [["gpt-4o", "gpt-4o"], ["claude-3-5-sonnet-20240620", None]]

    # Run the tests in parallel

    results = await asyncio.gather(*(test_case(test_class, model=a, deployment=b) for a, b in test_models))

    return results


check_plan_content = """
    Your task is to analyze the two plans given as a list of tuples, check if they relatively do the same.
    Answer with one word 'True' or 'False'
    Examples:
        #1 Example
            value a: [('query_orchestrator_agent', 'Retrieve the test stream's value data for 2020.'),
                    ('query_orchestrator_agent', 'Retrieve the expenses data for 2020.'),
                    ('calculation_agent', 'Subtract the expenses from the test streams value for the year 2020 to get the net value.')]

            value b: [('query_orchestrator_agent', 'Fetch the expenses data for the year 2020.'),
                    ('query_orchestrator_agent', 'Fetch the test streams value data for the year 2020.'),
                    ('calculation_agent', 'Subtract the expenses from the test stream's 2020 value to obtain the net value.')]

            result: True

        #2 Example
            value a: [('query_orchestrator_agent', 'Fetch the test streams value and expenses data for the year 2020.'),
                    ('calculation_agent', 'Subtract the expenses from the test streams value for the year 2020 to get the net value.')]

            value b: [('query_orchestrator_agent', 'Fetch the test streams value data for the year 2020.'),
                    ('query_orchestrator_agent', 'Fetch the expenses data for the year 2020.'),
                    ('calculation_agent', 'Subtract the expenses from the test streams value for the year 2020 to get the net value.')]

            result: True

        #3 Example
            value a: [('query_orchestrator_agent', 'Fetch the test streams value and emission data for the year 2020.'),
                    ('calculation_agent', 'Subtract the expenses from the test streams value for the year 2020 to get the net value.')]

            value b: [('query_orchestrator_agent', 'Fetch the test streams value data for the year 2020.'),
                    ('query_orchestrator_agent', 'Fetch the expenses data for the year 2021.'),
                    ('calculation_agent', 'Subtract the expenses from the test streams value for the year 2020 to get the net value.')]

            result: False
"""  # noqa: E501

check_calculation = """
    Your task is to analyze the two calculation result, check if they relatively the same.
    If we have a dicionary: check the variable names if they are similar.
    Check the floating values for at least 4 places.
    Don't look at the formatting, just make sure they are valid python objects.
    Answer only with one word 'True' or 'False', but without quotes.
    Examples:
        #1 Example
            value a: ["{'max': 1877120103.0798647, 'mean': 775275039.8735183, 'variance': 4.355705606555216e+17}"]

            value b: ["{\'max\': 1877120103.0798647, \'mean\': 775275039.8735183, \'variance\': 4.355705606555216e+17}"]

            result: True

        #2 Example
            value a: ["{'max': 1877120103.0798647, 'mean': 775275039.8735183, 'variance': 4.355705606555216e+17}"]

            value b: ["{'mx': 1901120103.08, 'var': 4.3557e+17, 'mean': 775275039.87351}"]

            result: False

        #2 Example
            value a: ["{'max': 1877120103.0798647, 'mean': 775275039.8735183, 'variance': 4.355705606555216e+17}"]

            value b: ["{'max': 1877120103.07986, 'std': 4.4557e+17, 'mean': 775275039.87351}"]

            result: False

        #3 Example
            value a: ["{'max': 1877120103.0798647, 'mean': 775275039.8735183, 'variance': 4.355705606555216e+17}"]

            value b: ["{'max': 1877120103.07986, 'var': 4.3557e+17, 'mean': 775275040}"]

            result: False
"""  # noqa: E501

check_research = """
You are an advanced language model designed to compare text retrievals.  
Key Rules and Restrictions:  
- Always answer with True or False  
- Avoid assumptions; rely solely on the information provided.

Examples:
#1 Example
value a: "The company tracks and reports its carbon footprint using a Greenhouse Gas Inventory Methodology that aligns
with the principles and guidelines set by the World Resources Institute and the World Business Council for Sustainable
Development’s Greenhouse Gas Protocol Initiative. Scope 1 emissions refer to direct emissions from on-site activities,
including the use of combustible fuels and other sources. Scope 2 emissions account for indirect emissions from the
generation of electricity or steam purchased off-site. Scope 3 emissions encompass additional indirect emissions
from activities such as product usage, waste management, commuting, and business travel. The company contracts an
independent verifier to assess and confirm the accuracy of the reported Scope 1 and 2 emissions on an annual basis.
Adjustments are made to account for changes in business structure, such as acquisitions or divestitures, to maintain
an accurate assessment of emission reduction progress."

value b: "The company measures and reports its carbon footprint by following a Greenhouse Gas Inventory
Methodology consistent with the principles and guidance of the World Resources Institute and the World Business
Council for Sustainable Development's Greenhouse Gas Protocol Initiative. Scope 1 emissions include direct
emissions from combustible fuels and other sources on-site. Scope 2 emissions include indirect emissions from
off-site electricity or steam production. Scope 3 emissions cover other indirect emissions from activities such
as product use, waste disposal, commuting, and business travel. The company engages an independent verifier to
verify reported Scope 1 and 2 emissions annually. Adjustments are made for structural changes in the business,
such as acquisitions or divestitures, to ensure accurate emission reduction progress."

result: True

#2 Example
value a: “The company monitors and reports its carbon emissions using a method that does not follow the World Resources
Institute or the Greenhouse Gas Protocol. Scope 1 emissions are primarily calculated based on assumptions rather than
direct data. Scope 2 emissions are rarely calculated and only account for a fraction of off-site energy use. Scope 3
emissions, which include product use and business travel, are ignored due to their complexity. The company has no
third-party verification process in place, and no adjustments are made for changes in the business structure, such as
acquisitions or divestitures.”

value b: “The company measures and reports its carbon footprint by following a Greenhouse Gas Inventory Methodology
consistent with the principles and guidance of the World Resources Institute and the World Business Council for
Sustainable Development’s Greenhouse Gas Protocol Initiative. Scope 1 emissions include direct emissions from
combustible fuels and other sources on-site. Scope 2 emissions include indirect emissions from off-site electricity or
steam production. Scope 3 emissions cover other indirect emissions from activities such as product use, waste disposal,
commuting, and business travel. The company engages an independent verifier to verify reported Scope 1 and 2 emissions
annually. Adjustments are made for structural changes in the business, such as acquisitions or divestitures, to ensure
accurate emission reduction progress.”


result: False
"""

check_same_meaning_prompt = """
    Compare the following phrases and return True if they have the similar meaning (i.e., convey the same idea with different wording or variable names are different but they convey the concept) or False if they do not. If False, briefly explain the difference.
"""  # noqa: E501


async def check_plan(expected_plan: list[tuple[str, str]], actual_plan: list[tuple[str, str]]) -> bool:
    return await call_gpt_comparison(expected_plan, actual_plan, check_plan_content)


async def check_calculations(expected_calculation: list[str], actual_calculation: list[str]) -> bool:
    return await call_gpt_comparison(expected_calculation, actual_calculation, check_calculation)


async def check_same_meaning(phrase_a: str, phrase_b: str) -> bool:
    return await call_gpt_comparison(value_a=phrase_a, value_b=phrase_b, prompt=check_same_meaning_prompt)


async def call_gpt_comparison(value_a, value_b, prompt: str) -> Coroutine[Any, Any, ChatCompletion]:
    llm = llm_factory_function(model=LLMModels.GPT_4O)
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=prompt),
        ChatMessage(role=MessageRole.USER, content=f"value a: {value_a},\n\nvalue b: {value_b}"),
    ]

    for i in range(0, MAX_RETRIES):
        response = (await llm.achat(messages=messages)).message.content

        if response in ["True", "False"]:
            return True if "True" else False

    raise ValueError("Something went wrong with the comparison")


def match_variables(source, target):
    matched = {}
    remaining_targets = set(target)

    for var in source:
        # Find the best match from the remaining targets
        match_result = process.extractOne(var, remaining_targets)
        if match_result is None:
            match = list(remaining_targets)[0]
        else:
            match, _, _ = match_result

        matched[var] = match
        remaining_targets.remove(match)

    return matched


# There might be some data loss on json formatting so have to apply heuristic comparison
def compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
    df1_comp = df1.astype(df2.dtypes)
    df2_comp = df2
    if "value" in df1.keys():
        df1_comp["value"] = df1.apply(lambda x: round(x["value"], 3), axis=1)
        df2_comp["value"] = df2.apply(lambda x: round(x["value"], 3), axis=1)
    diff = df1_comp.reset_index(drop=True) == df2_comp.reset_index(drop=True)
    return diff.all().all()


async def check_queried_datas(queried_data: dict[str, pd.DataFrame], expected_queried_data: dict[str, pd.DataFrame]):
    mapping = match_variables(queried_data.keys(), expected_queried_data.keys())
    if None in mapping.values():
        print(
            "Variable name couldn't be matched to an expected variable:"
            + f" {next((key for key, value in mapping.items() if value == None), None)}"
        )
        return False

    try:
        mapping = match_variables(queried_data.keys(), expected_queried_data.keys())
    except Exception:
        return False

    for key, value in queried_data.items():
        expected_var_name = mapping[key]
        expected_df = expected_queried_data[expected_var_name]
        if not compare_dataframes(value, expected_df):
            return False

    return True


async def check_researches(research_results: list[str], expected_research_results: list[str]) -> bool:
    if not await call_gpt_comparison("\n".join(research_results), "\n".join(expected_research_results), check_research):
        return False

    return True


async def check_response(response: tuple[str, str], expected: tuple[str, str]) -> bool:
    response_status = response[0]
    expected_status = expected[0]

    response_message = response[1]
    expected_message = expected[1]

    if not response_status == expected_status:
        raise ValueError("Response status different")

    # We only want to check need input messages
    if "NEED_INPUT" in expected_message and not await check_same_meaning(response_message, expected_message):
        raise ValueError("Need input messages' meanings aren't the same")

    return True
