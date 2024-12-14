from pathlib import Path

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, TextNode

from agentic_copilot.models.agents.query.document_query_tool import DocumentQueryTool
from agentic_copilot.models.utils.llm_utils import LLMModels


class ESGQueryEngine(DocumentQueryTool):
    def __init__(self, model: LLMModels = LLMModels.GPT_4O) -> None:
        super().__init__(document_name="research_material/walmart_esg", similarity_top_k=2, model=model)

    def _create_nodes(self) -> list[TextNode]:
        # Read the document content
        with open(Path(f"./data/{self.document_name}.txt"), "r") as f:
            text = f.read()

        # Split into sentences
        sentence_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
        nodes = sentence_parser.get_nodes_from_documents([Document(text=text)])

        # Add unique IDs to nodes
        for i, node in enumerate(nodes):
            node.id_ = f"walmart_esg_{i}"

        return nodes

    def query(self, query: str) -> str:
        return self.query_engine.query(
            f"""
            Answer the following question, but make short that the company name is excluded:
            {query}
        """
        )
