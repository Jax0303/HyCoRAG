from abc import ABC, abstractmethod
from typing import List, Dict, Any
from .retriever import BaseRetriever

class LLMClient(ABC):
    """
    Interface for LLM interactions.
    """
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

from dataclasses import dataclass
from typing import List, Optional

@dataclass
class RAGResult:
    answer: str
    retrieved_items: List[Any]
    metadata: Dict[str, Any] = None

    """
    Dummy LLM client for testing without API calls.
    """
    def generate(self, prompt: str) -> str:
        # Simple heuristic: if prompt contains "revenue", return a number
        if "revenue" in prompt.lower():
            return "100 million"
        return "Dummy answer based on context."

class BaseRAGPipeline(ABC):
    def __init__(self, retriever: BaseRetriever, llm: LLMClient):
        self.retriever = retriever
        self.llm = llm

    @abstractmethod
    def run(self, query: str) -> Any:
        pass

class BaselineRAGPipeline(BaseRAGPipeline):
    """
    Standard RAG pipeline.
    """
    def run(self, query: str, top_k: int = 3) -> RAGResult:
        # 1. Retrieve
        retrieved_items = self.retriever.retrieve(query, top_k=top_k)
        context = "\n\n".join([f"Table {item.id}: {item.text}" for item in retrieved_items])
        
        # 2. Generate
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        answer = self.llm.generate(prompt)
        
        return RAGResult(
            answer=answer, 
            retrieved_items=retrieved_items,
            metadata={"context_length": len(context)}
        )

from ..models.concept_distill import HybridConceptDistiller, HierarchicalConcepts
from ..models.concept_router import ConceptRouter, RoutedConcepts
import torch

from typing import Literal

from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass
import torch

from .retriever import BaseRetriever, RetrievedItem
from ..models.concept_distill import HybridConceptDistiller, HierarchicalConcepts
from ..models.concept_router import ConceptRouter, RoutedConcepts
from .llm_client import LLMClient, DummyLLMClient, LocalLLMClient
import torch

from typing import Literal

class HyCoRAGPipeline(BaseRAGPipeline):
    """
    HyCoRAG Pipeline: Retrieve -> Distill -> Route -> Generate.
    Supports ablation modes.
    """
    def __init__(self, 
                 retriever: BaseRetriever, 
                 llm_client: LLMClient,
                 distiller: HybridConceptDistiller,
                 router: ConceptRouter,
                 mode: Literal["baseline", "distill_only", "full"] = "full"):
        super().__init__(retriever, llm_client)
        self.distiller = distiller
        self.router = router
        self.mode = mode
        
        # Dummy query encoder for routing (if not provided by retriever)
        # In real impl, use the same encoder as retriever or separate one
        self.query_encoder = torch.nn.Linear(10, 768) # Dummy

    def run(self, query: str, top_k: int = 3) -> RAGResult:
        # 1. Standard Retrieval (Reuse Baseline Retriever)
        retrieved_items = self.retriever.retrieve(query, top_k=top_k)
        
        if self.mode == "baseline":
            # Fallback to standard baseline behavior (just use text)
            context = "\n\n".join([f"Table {item.id}: {item.text}" for item in retrieved_items])
            prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
            answer = self.llm.generate(prompt)
            return RAGResult(
                answer=answer, 
                retrieved_items=retrieved_items,
                metadata={"context_length": len(context)}
            )
        
        # 2. Hybrid Concept Distillation & Routing
        # For each retrieved table, we distill concepts and then route
        
        # Dummy query embedding for routing
        # In real impl, get from retriever or encode
        # Use router's input dimension
        router_dim = self.router.query_proj.in_features
        query_emb = torch.randn(1, router_dim) 
        
        selected_concepts_list = []
        
        for item in retrieved_items:
            # Simulate getting table content from item.text (which is flattened text)
            # In real impl, we'd fetch the structured object by ID
            
            # Distill
            # Extract structure from metadata if available
            if item.metadata and 'structure' in item.metadata:
                structure_input = item.metadata['structure']
            else:
                structure_input = {"text": item.text}
            
            hierarchical_concepts = self.distiller.distill(
                table_image=None, # TODO: Load image if available in metadata
                table_structure=structure_input, 
                context_text=item.text
            )
            
            if self.mode == "distill_only":
                # Use ALL concepts without routing (pruning)
                # Hack: create a RoutedConcepts with everything selected
                # For simplicity, we just pass everything.
                # In real impl, we might just concat all vectors.
                # Here we simulate "routing" that selects everything.
                # But RoutedConcepts expects tensors.
                # Let's just select top-k with very large k
                routed = self.router.route(query_emb, hierarchical_concepts, top_k=1000)
                selected_concepts_list.append(routed)
            else:
                # "full" mode: Route with limited top_k
                routed = self.router.route(query_emb, hierarchical_concepts, top_k=2)
                selected_concepts_list.append(routed)
            
        # 3. Context Construction
        # Serialize selected concepts into a compact string
        context_parts = []
        total_concepts = 0
        
        for i, routed in enumerate(selected_concepts_list):
            # Simple serialization for now
            # In real impl, we'd convert tensors back to text/tokens or use soft prompts
            
            # Count concepts
            n_sem = routed.semantic.size(0)
            n_str = routed.structural.size(0)
            n_ctx = routed.contextual.size(0)
            total_concepts += (n_sem + n_str + n_ctx)
            
            context_parts.append(f"Table {i+1} Concepts:")
            context_parts.append(f"- Semantic: {n_sem} items")
            context_parts.append(f"- Structural: {n_str} items")
            context_parts.append(f"- Contextual: {n_ctx} items")
            # Add original text snippet for dummy LLM to work
            context_parts.append(f"Original Snippet: {retrieved_items[i].text[:50]}...")

        context = "\n".join(context_parts)
        
        # Log stats (simulated logging)
        # print(f"[HyCoRAG] Selected {total_concepts} concepts from {len(retrieved_items)} tables.")
        
        # 4. Generation
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        answer = self.llm.generate(prompt)
        
        return RAGResult(
            answer=answer, 
            retrieved_items=retrieved_items,
            metadata={
                "context_length": len(context),
                "total_concepts": total_concepts,
                "concepts_per_table": total_concepts / max(1, len(retrieved_items))
            }
        )
