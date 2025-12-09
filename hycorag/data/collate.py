from typing import List
from .schemas import QAExample, QABatch

def collate_qa_examples(examples: List[QAExample]) -> QABatch:
    """
    Collate a list of QAExample into a QABatch.
    """
    questions = [ex.question for ex in examples]
    answers = [ex.answers for ex in examples]
    
    # Initially, contexts are just the provided documents
    contexts = [ex.documents for ex in examples]
    
    metadata = [ex.metadata for ex in examples]
    # Also include other fields in metadata if needed, or just keep them in QABatch if we extend it
    for i, ex in enumerate(examples):
        metadata[i]['qid'] = ex.qid
        metadata[i]['dataset_name'] = ex.dataset_name
        if ex.images:
            metadata[i]['images'] = ex.images

    return QABatch(
        questions=questions,
        contexts=contexts,
        answers=answers,
        metadata=metadata
    )
