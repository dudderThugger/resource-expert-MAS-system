from abc import ABC, abstractmethod
from pathlib import Path

from llama_index.core import Settings as llama_settings
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.schema import TextNode

from agentic_copilot.models.utils.llm_utils import (
    LLMModels,
    embedding_factory_function,
    llm_factory_function,
)


class DocumentQueryTool(ABC):
    """Class that ceates a query engine for a document that can be used for classes inheriting from this class"""

    def __init__(self, document_name: str, similarity_top_k: int = 5, model: LLMModels = LLMModels.GPT_4O):
        self.document_name = document_name
        self.similarity_top_k = similarity_top_k

        llama_settings.llm = llm_factory_function(model=model)
        llama_settings.embed_model = embedding_factory_function()

        self.query_engine = self._create_engine()

    @abstractmethod
    def _create_nodes(self) -> list[TextNode]:
        pass

    def _create_storage_context(self) -> VectorStoreIndex:
        nodes = self._create_nodes()

        index = VectorStoreIndex(nodes=nodes, embed_model=embedding_factory_function())
        index.storage_context.persist(Path(f"data/{self.document_name}"))

        return index

    def _build_index(self) -> VectorStoreIndex:
        index = None

        if not Path(f"./data/{self.document_name}/index_store.json").exists():
            index = self._create_storage_context()

        else:
            storage_context = StorageContext.from_defaults(persist_dir=Path(f"data/{self.document_name}"))
            index = load_index_from_storage(storage_context)

        return index

    def _create_engine(self) -> BaseQueryEngine:
        return self._build_index().as_query_engine(similarity_top_k=self.similarity_top_k)
