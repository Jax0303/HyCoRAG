from typing import Dict, Any
from ..rag.pipeline import BaseRAGPipeline
from ..data.datasets import TableQADataset

class ExperimentRunner:
    """
    Runs experiments and evaluates results.
    """
from .metrics import calculate_em, calculate_f1, calculate_hit_at_k
import numpy as np

class ExperimentRunner:
    """
    Runs experiments and evaluates results.
    """
    def run_baseline_experiment(self, pipeline: BaseRAGPipeline, dataset: TableQADataset, top_k: int = 3) -> Dict[str, float]:
        """
        Run evaluation on the dataset.
        """
        print(f"Running experiment on {len(dataset)} samples...")
        
        em_scores = []
        f1_scores = []
        hit_scores = []
        
        for i in range(len(dataset)):
            item = dataset[i]
            question = item.question
            gold_answer = item.answer
            gold_table_id = item.table_id
            
            result = pipeline.run(question, top_k=top_k)
            
            # Metrics
            em = calculate_em(result.answer, gold_answer)
            f1 = calculate_f1(result.answer, gold_answer)
            
            retrieved_ids = [r.id for r in result.retrieved_items]
            hit = calculate_hit_at_k(retrieved_ids, gold_table_id)
            
            em_scores.append(em)
            f1_scores.append(f1)
            hit_scores.append(hit)
            
            if i % 10 == 0:
                print(f"Processed {i}/{len(dataset)} samples.")
                
        metrics = {
            "avg_em": float(np.mean(em_scores)),
            "avg_f1": float(np.mean(f1_scores)),
            f"hit@{top_k}": float(np.mean(hit_scores))
        }
        return metrics
