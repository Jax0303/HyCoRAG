from torch.utils.data import DataLoader
from ..data.datasets import HotpotQADataset, NQOpenDataset, RAGIGBenchDataset
from ..data.collate import collate_qa_examples

def get_hotpot_dataloader(split: str = "train", batch_size: int = 32, shuffle: bool = False) -> DataLoader:
    dataset = HotpotQADataset.from_hf(split=split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_qa_examples)

def get_nq_dataloader(split: str = "train", batch_size: int = 32, shuffle: bool = False) -> DataLoader:
    dataset = NQOpenDataset.from_hf(split=split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_qa_examples)

def get_rag_igbench_dataloader(split: str = "train", batch_size: int = 32, shuffle: bool = False) -> DataLoader:
    dataset = RAGIGBenchDataset.from_hf(split=split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_qa_examples)
