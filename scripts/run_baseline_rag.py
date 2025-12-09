import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from hycorag.data.datasets import TableQADataset
from hycorag.rag.pipeline import BaselineRAGPipeline, DummyLLMClient
from hycorag.rag.retriever import BaselineRetriever
from hycorag.evaluation.experiment_runner import ExperimentRunner

def create_toy_data(path: str):
    data = [
        {
            "id": "table1",
            "flattened_text": "Company A Revenue 2023: $100M. Profit: $10M.",
            "qa_pairs": [
                {"question": "What is the revenue of Company A in 2023?", "answer": "$100M"},
                {"question": "What is the profit of Company A?", "answer": "$10M"}
            ]
        },
        {
            "id": "table2",
            "flattened_text": "Company B Revenue 2023: $200M. Profit: $20M.",
            "qa_pairs": [
                {"question": "What is the revenue of Company B?", "answer": "$200M"}
            ]
        },
        {
            "id": "table3",
            "flattened_text": "Company C Revenue 2023: $300M. Profit: $30M.",
            "qa_pairs": []
        }
    ]
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Created toy data at {path}")

def main():
    print("Initializing Baseline RAG Pipeline...")
    
    # 1. Prepare Data
    data_path = "toy_data.json"
    create_toy_data(data_path)
    dataset = TableQADataset.from_json(data_path)
    
    # 2. Setup components
    llm = DummyLLMClient()
    retriever = BaselineRetriever(embedding_dim=128)
    
    # 3. Indexing
    print("Indexing tables...")
    tables = dataset.get_all_tables()
    retriever.index(tables)
    
    pipeline = BaselineRAGPipeline(retriever=retriever, llm=llm)
    
    # 4. Run Experiment
    runner = ExperimentRunner()
    metrics = runner.run_baseline_experiment(pipeline, dataset, top_k=2)
    
    print("\n=== Experiment Results ===")
    print(json.dumps(metrics, indent=2))
    
    # Clean up
    if os.path.exists(data_path):
        os.remove(data_path)
    print("Done.")

if __name__ == "__main__":
    main()
