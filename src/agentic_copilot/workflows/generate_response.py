from llama_index.core import PromptTemplate

from agentic_copilot.models.utils.agents_util import AgentsState
from agentic_copilot.models.utils.llm_utils import LLMModels, llm_factory_function

GENERATE_ANSWER_PROMPT_TEMPLATE = PromptTemplate(
    """
    YOU ARE A MULTI-AGENT SYSTEM SPECIALIZED IN GENERATING CONTEXTUAL ANSWERS BY SYNTHESIZING MULTIPLE CONVERSATION STATES. CREATE A RESPONSE FOR THE USER USING THE FOLLOWING PARAMETERS:

    - **BASE UTTERANCE**: {base_utterance}
    - **EXECUTION PLAN**: {execution_plan}
    - **CHAT HISTORY SUMMARY**: {history_summary}
    - **QUERY RESULTS (DataFrame)**: {queried_data}
    - **CALCULATED DATA**: {calculated_data}

    ###INSTRUCTIONS###
    1. REVIEW EACH PARAMETER TO UNDERSTAND THE FULL CONTEXT.
    2. USE THE EXECUTION PLAN TO GUIDE YOUR RESPONSE STRUCTURE.
    3. SYNTHESIZE INSIGHTS FROM THE CHAT HISTORY, QUERY RESULTS, AND CALCULATED DATA TO ANSWER THE BASE UTTERANCE CLEARLY AND CONCISELY.
    4. PROVIDE A FINAL ANSWER THAT DIRECTLY ADDRESSES THE USER’S NEEDS, HIGHLIGHTING KEY FINDINGS OR RELEVANT PATTERNS.

    ###WHAT NOT TO DO###
    - DO NOT INCLUDE RAW OR IRRELEVANT DATA WITHOUT CONTEXT.
    - AVOID REDUNDANT DETAILS THAT DO NOT DIRECTLY ENHANCE THE ANSWER.
                                                 
    ###EXAMPLE OUTPUT FORMAT###
    - YOU SHOULD REPLACE THE PARTS BETWEEN THE TAGS '<>'

    "Based on the data and calculations performed, here are the insights you requested:
                                                 
    <results structured and seperated clearly>

    These values reflect <if relevant the period it was calculated on and the sites> by using <the used datastreams exact name>. If you need further details or additional analysis, feel free to ask!"
"""  # noqa: 501
)


async def generate_response(state: AgentsState) -> str:
    prompt = GENERATE_ANSWER_PROMPT_TEMPLATE.format(
        base_utterance=state.base_utterance,
        execution_plan=state.plan,
        history_summary="\n".join(state.chat_history),
        queried_data=state.queried_data,
        calculated_data=state.calculation_results,
    )
    return (await llm_factory_function(model=LLMModels.GPT_4O).acomplete(prompt)).text


REQUEST_FOR_INPUT_PROMPT = PromptTemplate(
    """
    YOU ARE A MULTI-AGENT SYSTEM DESIGNED TO REQUEST ADDITIONAL INFORMATION FROM THE USER TO COMPLETE A TASK, BASED ON EXISTING PARAMETERS AND CONTEXT. YOUR RESPONSE SHOULD BE CLEAR AND FOCUSED ON OBTAINING THE NECESSARY INFORMATION.

    ### PARAMETERS ###

    - **MESSAGE**: {message}
    - **BASE UTTERANCE**: {base_utterance}
    - **EXECUTION PLAN**: {execution_plan}
    - **CHAT HISTORY SUMMARY**: {chat_history_summary}
    - **QUERY RESULTS**: {query_results}
    - **CALCULATED DATA**: {calculated_data}

    ### INSTRUCTIONS ###

    1. **ANALYZE** each parameter to ensure you fully understand the current state of the conversation.
    2. **IDENTIFY** any gaps in information based on the message parameter and the requirements outlined in the execution plan.
    3. **FORMULATE** a concise and respectful question or series of questions aimed at obtaining the missing details from the user.
    4. **REFERENCE** relevant parts of the chat history, query results, or calculated data, if necessary, to provide context and make your request more specific.
    5. **CLARIFY** why the additional information is needed to move forward with the execution plan and enhance the system's response quality.

    ### WHAT NOT TO DO ###

    - DO NOT ASK FOR INFORMATION THAT HAS ALREADY BEEN PROVIDED BY THE USER IN THE CHAT HISTORY.
    - AVOID VAGUE OR GENERIC QUESTIONS; MAKE YOUR REQUEST SPECIFIC TO THE CURRENT CONTEXT.
    - DO NOT INCLUDE RAW DATA OR UNPROCESSED QUERY RESULTS WITHOUT EXPLANATION.
    - AVOID REDUNDANT REQUESTS OR REQUESTING DATA THAT IS IRRELEVANT TO THE EXECUTION PLAN.

    ### EXAMPLE OUTPUT FORMAT ###
    - YOU SHOULD REPLACE THE PARTS BETWEEN THE TAGS '<>'
                                                 
    "To proceed with <execution_plan>, could you please provide further details on <specific information needed from 'message'>? 
        Based on our previous discussion, it appears that additional input regarding <specific data gap> is essential. 
        Our current findings (<something about the query results> and <something about the calculated results>}) suggest that this information will help us <explain how it will aid in achieving the execution plan objective>. 
    Thank you!"
"""  # noqa: 501
)


async def generate_request_input(message: str, state: AgentsState):
    prompt = REQUEST_FOR_INPUT_PROMPT.format(
        message=message,
        base_utterance=state.base_utterance,
        execution_plan=state.plan,
        chat_history_summary="\n".join(state.chat_history),
        query_results=state.queried_data,
        calculated_data=state.calculation_results,
    )

    return (await llm_factory_function(LLMModels.GPT_4O).acomplete(prompt)).text
