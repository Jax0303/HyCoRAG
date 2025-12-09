"""
Debug script to trace metadata through the full pipeline.
"""
import sys
sys.path.insert(0, '/home/user/HyCoRAG')

from hycorag.data.datasets import RealHiTBenchDataset
from hycorag.rag.retriever import BaselineRetriever
from hycorag.rag.pipeline import HyCoRAGPipeline, DummyLLMClient
from hycorag.models.table_encoder import TableEncoder
from hycorag.models.concept_distill import HybridConceptDistiller
from hycorag.models.concept_router import ConceptRouter

# Load dataset
dataset = RealHiTBenchDataset.from_local("RealHiTBench", max_samples=2)
print(f"Loaded {len(dataset)} samples")

# Check first sample metadata
sample = dataset[0]
print(f"\n=== Sample 0 Metadata ===")
print(f"Has 'structure' key: {'structure' in sample.metadata}")
if 'structure' in sample.metadata:
    print(f"Number of cells in structure: {len(sample.metadata['structure']['cells'])}")

# Build corpus like experiments.py does
corpus = {}
corpus_metadata = {}

for i, item in enumerate(dataset):
    for j, doc in enumerate(item.documents):
        doc_id = f"doc_{item.qid}_{j}"
        corpus[doc_id] = doc
        if item.metadata and 'structure' in item.metadata:
            corpus_metadata[doc_id] = item.metadata
            print(f"Stored metadata for {doc_id}")

print(f"\n=== Corpus ===")
print(f"Total docs: {len(corpus)}")
print(f"Docs with metadata: {len(corpus_metadata)}")

# Initialize retriever
retriever = BaselineRetriever()
retriever.index(corpus)
retriever._corpus_metadata = corpus_metadata

print(f"\n=== Retriever State ===")
print(f"Has _corpus_metadata: {hasattr(retriever, '_corpus_metadata')}")
if hasattr(retriever, '_corpus_metadata'):
    print(f"Metadata keys: {list(retriever._corpus_metadata.keys())}")

# Test retrieval
query = sample.question
retrieved = retriever.retrieve(query, top_k=1)

print(f"\n=== Retrieved Items ===")
for item in retrieved:
    print(f"ID: {item.id}")
    print(f"Text (first 100): {item.text[:100]}")
    print(f"Metadata keys: {item.metadata.keys() if item.metadata else 'None'}")
    if item.metadata and 'structure' in item.metadata:
        print(f"Structure cells: {len(item.metadata['structure']['cells'])}")
    else:
        print("NO STRUCTURE IN METADATA!")
