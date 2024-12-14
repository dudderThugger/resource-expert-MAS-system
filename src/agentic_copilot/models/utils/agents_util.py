import json
import logging
from enum import Enum
from pathlib import Path
from typing import Optional, TextIO

from pandas import DataFrame
from pydantic import FilePath


class Speaker(str, Enum):
    CALCULATION = "calculation_agent"
    DATASTREAM_QUERY = "datastream_query_agent"
    INVOICE_QUERY = "invoice_query_agent"
    ORCHESTRATOR = "orchestrator_agent"
    PLANNING = "planning_agent"
    QUERY_ORCHESTRATOR = "query_orchestrator_agent"
    RESEARCH_AGENT = "research_agent"


class AgentsState(object):
    def __init__(self, user_id) -> None:
        self.user_id = user_id
        self.base_utterance = None
        self.plan = []
        self.research_results: list[str] = []
        self.queried_data: dict[str, DataFrame] = {}
        self.chat_history = []
        self.current_step: Optional[int] = None
        self.calculation_results: list[str] = []

    def get_current_step(self) -> tuple[str, str]:
        if self.current_step is None:
            return ()

        return self.plan[self.current_step]

    def advance_step(self) -> tuple[str, str]:
        step = self.plan[self.current_step]
        self.current_step += 1
        return step

    def has_more_step(self) -> bool:
        return self.current_step < len(self.plan)

    def modify_current_step(self, modified_instruction: str) -> str:
        step = self.plan[self.current_step]
        speaker_of_step = step[0]
        self.plan[self.current_step] = (speaker_of_step, modified_instruction)

        return self.plan[self.current_step]

    def get_state_string(self) -> str:
        return f"""{{
                user_id: {self.user_id},
                plan: {self.plan},
                research_results: {self.research_results},
                queried_data (first 2 rows of each DataFrame): {'  -----------   '.join([f"Query {i}: {self.queried_data[data].to_json()}" for i, data in enumerate(self.queried_data)])},
                chat_history: {self.chat_history},
                current_step: {self.current_step},
                calculation_result: {self.calculation_results}
            }}"""  # noqa: E501

    def get_json(self) -> str:
        queried_data_json = dict([(var, df.to_dict()) for var, df in self.queried_data.items()])
        json_output = {
            "client_id": self.user_id,
            "base_utterance": self.base_utterance,
            "plan": self.plan,
            "research_results": self.research_results,
            "queried_data": queried_data_json,
            "chat_history": self.chat_history,
            "current_step": self.current_step,
            "calculation_results": self.calculation_results,
        }

        return json.dumps(json_output)


def load_state_from_json(file: FilePath) -> AgentsState:
    json_input: dict = {}

    with open(file, "r") as f:
        json_input = json.load(f)

    state = AgentsState(json_input["client_id"])
    for key, value in json_input["queried_data"].items():
        state.queried_data[key] = DataFrame(value)

    for field, value in json_input.items():
        if field != "queried_data":
            setattr(state, field, value)

    return state


def get_logger(name: str, stream_output: TextIO, file_output_location: str = None) -> logging.Logger:
    if name in logging.Logger.manager.loggerDict:
        return logging.getLogger(name)
    else:
        logger = logging.getLogger(name)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        if file_output_location is not None:
            file_handler = logging.FileHandler(Path(file_output_location))
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.INFO)
            logger.addHandler(file_handler)

        console_handler = logging.StreamHandler(stream_output)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)

        logger.info(f"New logger created for {name}")
        return logger


def eval_response(agent_response: str) -> tuple[str, str]:
    try:
        striped = agent_response.split(",", maxsplit=1)
        status = striped[0].replace("(", "", 1).replace("'", "", 1)[::-1].replace("'", "", 1)[::-1]
        message = striped[1][::-1].replace(")", "", 1)[::-1].replace("'", "", 1)[::-1].replace("'", "", 1)[::-1]
    except Exception:
        raise ValueError("Agent didn't use its return_direct tools")

    return status, message
