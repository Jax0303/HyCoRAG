import datasets as hf_datasets

def main():
    print("Loading RAG-IGBench...")
    ds = hf_datasets.load_dataset("Muyi13/RAG-IGBench", split="train", streaming=True)
    
    print("Inspecting first 5 samples...")
    for i, item in enumerate(ds):
        if i >= 5: break
        print(f"Sample {i}:")
        print(f"Keys: {item.keys()}")
        print(f"Query: {item['query']}")
        # print(f"Language: {item.get('language', 'N/A')}")
        print("-" * 20)

if __name__ == "__main__":
    main()
