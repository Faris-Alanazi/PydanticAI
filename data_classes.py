from dataclasses import dataclass

@dataclass
class RAGResult:
    """
    Represents a structured result for the RAG query.
    """
    chunk: str
    similarity: float
