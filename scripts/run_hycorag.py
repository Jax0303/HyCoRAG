import sys
import os
import torch
import argparse

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hycorag.rag.retriever import BaselineRetriever
from hycorag.rag.llm_client import DummyLLMClient, LocalLLMClient
from hycorag.models.concept_distill import HybridConceptDistiller
from hycorag.models.concept_router import ConceptRouter
from hycorag.evaluation.experiments import (
    run_text_qa_experiment,
    run_table_mm_experiment,
    run_hierarchical_experiment
)

def main():
    parser = argparse.ArgumentParser(description="Run HyCoRAG Experiments")
    parser.add_argument("--stage", type=str, required=True,
                        choices=["text_qa", "table_mm", "table_hier"],
                        help="Experiment stage")
    parser.add_argument("--mode", type=str, default="baseline",
                        choices=["baseline", "distill_only", "full"],
                        help="Ablation mode")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (hotpotqa, nq_open, mmtab, comtqa, realhitbench, rag_igbench)")
    parser.add_argument("--max_samples", type=int, default=10,
                        help="Max samples to evaluate")
    parser.add_argument("--use_dummy_llm", action="store_true",
                        help="Use dummy LLM instead of real model (for testing)")

    args = parser.parse_args()

    # Initialize models
    print("Initializing models...")
    # 1. Retriever (Mock index for now if needed, or rely on pipeline to mock it)
    retriever = BaselineRetriever(embedding_dim=768)

    # 2. LLM
    if args.use_dummy_llm:
        print("Using DummyLLMClient (no real generation)")
        llm = DummyLLMClient()
    else:
        print("Loading LocalLLM (Qwen2.5-3B)...")
        llm = LocalLLMClient(model_name="Qwen/Qwen2.5-3B-Instruct")

    # 3. HyCoRAG components
    distiller = HybridConceptDistiller(hidden_dim=768)
    router = ConceptRouter(hidden_dim=768)

    metrics = {}

    if args.stage == "text_qa":
        metrics = run_text_qa_experiment(
            args.dataset, args.mode, retriever, llm, distiller, router, args.max_samples
        )
    elif args.stage == "table_mm":
        # table_mm covers mmtab, rag_igbench
        metrics = run_table_mm_experiment(
            args.dataset, args.mode, retriever, llm, distiller, router, args.max_samples
        )
    elif args.stage == "table_hier":
        metrics = run_hierarchical_experiment(
            args.dataset, args.mode, retriever, llm, distiller, router, args.max_samples
        )
    else:
        print(f"Unknown stage: {args.stage}")
        return

    # Pretty Print
    print("\n" + "="*30)
    print(f"RESULTS | Stage: {args.stage} | Dataset: {args.dataset} | Mode: {args.mode}")
    print("="*30)
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print("="*30)

if __name__ == "__main__":
    main()
