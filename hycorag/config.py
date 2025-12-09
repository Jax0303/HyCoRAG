import os
from dataclasses import dataclass

@dataclass
class HyCoRAGConfig:
    """
    Configuration for HyCoRAG system.
    """
    model_name: str = "bert-base-uncased"
    max_seq_length: int = 512
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    
    # Retrieval settings
    top_k: int = 5
    
    # Paths
    data_dir: str = "data"
    output_dir: str = "output"

    def __post_init__(self):
        # TODO: Load from environment variables or config file if needed
        pass
