from datasets import get_dataset_config_names
try:
    configs = get_dataset_config_names("Muyi13/RAG-IGBench")
    print(configs)
except Exception as e:
    print(e)
