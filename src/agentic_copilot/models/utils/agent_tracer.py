import json
from llama_index.core.callbacks.base_handler import BaseCallbackHandler

from typing import Any, Dict, List, Optional
from llama_index.core.callbacks.schema import CBEventType
import requests

from colorama import Fore, Style
from agentic_copilot.models.utils.llm_utils import LLMModels

blue = "\033[1;34m"
yellow = "\033[33m"
end = "\033[0m"


blue = "\033[1;34m"
yellow = "\033[33m"
end = "\033[0m"


response = requests.get("http://0.0.0.0:4000/v1/model/info")
json_prices = json.loads(response.content.decode("utf-8"))
PRICES = {
    model["model_name"]: {
        "output_price": model["model_info"]["output_price"],
        "input_price": model["model_info"]["input_price"],
    }
    for model in json_prices["data"]
}


class AgentTracer(BaseCallbackHandler):
    def __init__(self, model, cli_print: bool = True):
        super().__init__([], [])
        self.model = model
        self.cli_print = cli_print
        self.messages: dict[str, List] = {}
        self.input_tokens = 0
        self.output_tokens = 0
        self.llm_calls = 0

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        """Run when an event starts and return id of event."""
        messages = []
        if event_id not in self.messages:
            self.messages[event_id] = []
        if event_type == CBEventType.LLM:
            messages = []
            if "messages" in payload.keys():
                for message in payload["messages"]:
                    messages.append({"role": message.role.value, "content": message.content})
                self.messages[event_id].append({"type": "llm_input", "content": messages})
                if self.cli_print:
                    self._print_message(event_id=event_id, event_type=event_type, messages=messages, color=Fore.YELLOW)

        if event_type == CBEventType.FUNCTION_CALL:
            if "tool" in payload.keys():
                self.messages[event_id].append(
                    {
                        "type": "function_call",
                        "content": {"function_name": payload["tool"].name, "args": payload["function_call"]},
                    }
                )
                if self.cli_print:
                    self._print_message(event_id=event_id, event_type=event_type, messages=messages, color=Fore.GREEN)
        self.messages[event_id].extend(messages)

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Run when an event ends."""
        messages = []
        if event_type == CBEventType.LLM:
            choices = []
            if "response" in payload.keys():
                for choice in payload["response"].raw.choices:
                    choices.append({"role": choice.message.role, "content": choice.message.content})
                    if "assistant" == str(choice.message.role) and choice.message.tool_calls is not None:
                        for tool_call in choice.message.tool_calls:
                            choices.append({"tool_call_args": tool_call.function.arguments})
                messages.append({"type": "llm_output", "content": choices})
                if self.cli_print:
                    self._print_message(event_id=event_id, event_type=event_type, messages=messages, color=Fore.GREEN)

                self.input_tokens += payload["response"].raw.usage.prompt_tokens
                self.output_tokens += payload["response"].raw.usage.completion_tokens
        if event_type == CBEventType.FUNCTION_CALL:
            messages.append({"type": "function_response", "content": payload})
            if self.cli_print:
                self._print_message(event_id=event_id, event_type=event_type, messages=messages, color=Fore.GREEN)
        self.messages[event_id].extend(messages)

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """Run when an overall trace is launched."""
        pass

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """Run when an overall trace is exited."""
        pass

    def _print_message(self, event_id, event_type, messages, color):
        print(Fore.YELLOW + event_type + " " + event_id)
        print(color)

        def format_value(value, indent_level=2):
            indent = " " * indent_level
            if isinstance(value, dict):
                formatted = ["{"]
                for k, v in value.items():
                    formatted.append(f"{indent}{k.capitalize()}: {format_value(v, indent_level + 2)}")
                formatted.append(" " * (indent_level - 2) + "}")
                return "\n".join(formatted)
            elif isinstance(value, list):
                formatted = ["["]
                for item in value:
                    formatted.append(f"{indent}- {format_value(item, indent_level + 2)}")
                formatted.append(" " * (indent_level - 2) + "]")
                return "\n".join(formatted)
            else:
                return str(value)

        if not isinstance(messages, list):
            return "Input is not a list of messages."

        formatted_output = []
        for i, message in enumerate(messages, start=1):
            formatted_output.append("Message:\n")
            if isinstance(message, dict):
                for key, value in message.items():
                    formatted_output.append(f"{key.capitalize()}:\n{format_value(value, 4)}")
            else:
                formatted_output.append("  Invalid message format (not a dictionary).")
            formatted_output.append("")  # Add a blank line between messages

        print("\n".join(formatted_output))
        print(Style.RESET_ALL)

    @property
    def price(self) -> float:
        input_price = PRICES[self.model.value]["input_price"]
        output_price = PRICES[self.model.value]["output_price"]
        return (input_price * self.input_tokens + output_price * self.output_tokens) * 0.000001
