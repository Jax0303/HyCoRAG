"""
Debug script to trace concept generation pipeline.
"""
import sys
sys.path.insert(0, '/home/user/HyCoRAG')

from hycorag.data.datasets import RealHiTBenchDataset
from hycorag.models.table_encoder import TableEncoder
from hycorag.models.concept_distill import HybridConceptDistiller

# Load 1 sample
dataset = RealHiTBenchDataset.from_local("RealHiTBench", max_samples=1)
print(f"\n=== Loaded {len(dataset)} samples ===")

if len(dataset) > 0:
    sample = dataset[0]
    print(f"\nQuestion: {sample.question}")
    print(f"Answer: {sample.answers}")
    print(f"Document (first 200 chars): {sample.documents[0][:200]}")
    print(f"\nMetadata keys: {sample.metadata.keys()}")
    
    # Check structure
    if 'structure' in sample.metadata:
        struct = sample.metadata['structure']
        print(f"\nStructure keys: {struct.keys()}")
        print(f"Number of cells: {len(struct.get('cells', []))}")
        if struct.get('cells'):
            print(f"First cell: {struct['cells'][0]}")
    
    # Test TableEncoder
    print("\n=== Testing TableEncoder ===")
    encoder = TableEncoder(hidden_dim=768)
    
    # Pass structure to encoder
    structure_input = sample.metadata.get('structure', {"text": sample.documents[0]})
    embeddings = encoder.encode_hybrid_table(
        table_image=None,
        table_structure=structure_input,
        context_text=sample.documents[0]
    )
    
    print(f"Cell embeddings shape: {embeddings['cell_embeddings'].shape}")
    print(f"Row embeddings shape: {embeddings['row_embeddings'].shape}")
    print(f"Col embeddings shape: {embeddings['col_embeddings'].shape}")
    
    # Test Distiller
    print("\n=== Testing Distiller ===")
    distiller = HybridConceptDistiller(hidden_dim=768, table_encoder=encoder)
    concepts = distiller.distill(
        table_image=None,
        table_structure=structure_input,
        context_text=sample.documents[0]
    )
    
    print(f"Semantic concepts: {concepts.semantic.shape}")
    print(f"Structural concepts: {concepts.structural.shape}")
    print(f"Contextual concepts: {concepts.contextual.shape}")
    print(f"Total concepts: {concepts.semantic.shape[0] + concepts.structural.shape[0] + concepts.contextual.shape[0]}")
