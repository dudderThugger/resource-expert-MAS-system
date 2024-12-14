import re
from pathlib import Path

import pandas as pd
from llama_index.core.schema import TextNode

from agentic_copilot.models.agents.query.document_query_tool import DocumentQueryTool
from agentic_copilot.models.utils.llm_utils import LLMModels


class ClientDataStreamMatchingEngine(DocumentQueryTool):
    def __init__(self, client_id: int) -> None:
        super().__init__(document_name=f"datastream_name_indexes/client{client_id}", model=LLMModels.GPT_4O)
        self.client_id = client_id
        full_df = pd.read_csv(Path("data/datastreams_full_synth.csv"), encoding="ISO-8859-1")
        self.ds_names = full_df[full_df["client_id"] == self.client_id][["data_stream"]].drop_duplicates().reset_index()

    def _create_nodes(self) -> list[TextNode]:
        # Making nodes to embed and index for the VectorStoreIndex
        nodes_ds = []

        for datastream in enumerate(self.ds_names.iloc[:, 1:].to_dict(orient="records")):
            text = re.sub("[{}']", "", str(datastream[1]))
            nodes_ds.append(TextNode(text=text, id_=f"datastream_client{self.client_id}_{datastream[0]}"))

        return nodes_ds

    def match_datastream(self, data_stream_str: str) -> list[str]:
        response = str(
            self.query_engine.query(
                f"""
                    The datastream you should find: {data_stream_str}
                    Return as many as you think is relevant at most 10.
                    Answer with just the indexes and nothing else in list format: ['Water Usage', 'Energy Consumption', ...] that can be interpreted by python eval() function.
                    DONT ANSWER WITH TEXT EVEN IF YOU DIDNT FIND ANYTHING
                """  # noqa: E501
            )
        )
        matches = eval(response)

        if data_stream_str in matches:
            return [data_stream_str]
        else:
            return sorted(matches)
