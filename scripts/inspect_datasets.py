import argparse
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hycorag.data.datasets import HotpotQADataset, NQOpenDataset, RAGIGBenchDataset

def main():
    parser = argparse.ArgumentParser(description="Inspect HyCoRAG Datasets")
    parser.add_argument("--dataset", type=str, required=True, choices=["hotpotqa", "nq_open", "rag_igbench"], help="Dataset to inspect")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (train/validation/etc)")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of samples to print")
    
    args = parser.parse_args()
    
    print(f"Loading {args.dataset} ({args.split})...")
    
    if args.dataset == "hotpotqa":
        dataset = HotpotQADataset.from_hf(split=args.split)
    elif args.dataset == "nq_open":
        dataset = NQOpenDataset.from_hf(split=args.split)
    elif args.dataset == "rag_igbench":
        dataset = RAGIGBenchDataset.from_hf(split=args.split)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
        
    print(f"Loaded {len(dataset)} samples.")
    print("-" * 50)
    
    for i in range(min(args.num_samples, len(dataset))):
        ex = dataset[i]
        print(f"Sample {i}:")
        print(f"  QID: {ex.qid}")
        print(f"  Question: {ex.question}")
        print(f"  Answers: {ex.answers}")
        print(f"  Num Documents: {len(ex.documents)}")
        if ex.documents:
            print(f"  First Document Preview: {ex.documents[0][:100]}...")
        if ex.images:
            print(f"  Images: Yes ({len(ex.images)} docs have images)")
            if ex.images[0]:
                print(f"  First Image URL: {ex.images[0][0]}")
        else:
            print(f"  Images: No")
        print(f"  Metadata: {ex.metadata}")
        if args.dataset == "rag_igbench":
             # Hack to access raw item to check for other fields
             pass 
        print("-" * 50)

if __name__ == "__main__":
    main()
