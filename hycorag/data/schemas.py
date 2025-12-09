from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class QAExample:
    """
    Unified schema for QA examples.
    """
    qid: str
    dataset_name: str  # "hotpotqa", "nq_open", "rag_igbench"
    question: str
    answers: List[str]  # List of valid answers
    documents: List[str]  # List of document texts (evidence/context)
    images: Optional[List[List[str]]] = None  # List of image URLs per document (for RAG-IGBench)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Extra info (split, category, etc.)

@dataclass
class QABatch:
    """
    Batch of QA examples.
    """
    questions: List[str]
    contexts: List[List[str]]  # Initial contexts (from dataset)
    answers: List[List[str]]
    metadata: List[Dict[str, Any]]
    # TODO: Add fields for retrieved contexts later
