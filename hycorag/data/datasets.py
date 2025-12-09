from typing import List, Dict, Any, Optional
from torch.utils.data import Dataset
import json
import os
import glob
from dataclasses import dataclass

# Hugging Face Datasets
import datasets as hf_datasets

from .schemas import QAExample

# --- Legacy Support (for toy data) ---
@dataclass
class QASample:
    question: str
    answer: str
    table_id: str
    table_text: str

class TableQADataset(Dataset):
    """
    Legacy Dataset for simple Table QA tasks (internal usage).
    """
    def __init__(self, samples: List[QASample]):
        self.samples = samples

    @classmethod
    def from_json(cls, path: str) -> 'TableQADataset':
        """Load from simple JSON format."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        samples = []
        for item in data:
            table_id = item.get("id")
            table_text = item.get("flattened_text", "")
            for qa in item.get("qa_pairs", []):
                samples.append(QASample(
                    question=qa.get("question", ""),
                    answer=qa.get("answer", ""),
                    table_id=table_id,
                    table_text=table_text
                ))
        return cls(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> QASample:
        return self.samples[idx]

    def get_all_tables(self) -> Dict[str, str]:
        tables = {}
        for s in self.samples:
            tables[s.table_id] = s.table_text
        return tables

def collate_fn(batch: List[QASample]) -> Dict[str, Any]:
    """Collate function for batching."""
    return {
        "questions": [b.question for b in batch],
        "answers": [b.answer for b in batch],
        "table_ids": [b.table_id for b in batch],
        "table_texts": [b.table_text for b in batch]
    }

# --- Unified Data Layer ---

class BaseQADataset(Dataset):
    """
    Base class for QA datasets using the unified QAExample schema.
    """
    def __init__(self, examples: List[QAExample]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx) -> QAExample:
        return self.examples[idx]

class HotpotQADataset(BaseQADataset):
    @classmethod
    def from_hf(cls, split: str = "train", max_samples: Optional[int] = None) -> 'HotpotQADataset':
        print(f"Loading HotpotQA (fullwiki) split={split}...")
        try:
            ds = hf_datasets.load_dataset("hotpot_qa", "fullwiki", split=split)
        except Exception as e:
            print(f"Error loading HotpotQA: {e}")
            return cls([])

        if max_samples:
            ds = ds.select(range(min(len(ds), max_samples)))

        examples = []
        for item in ds:
            # HotpotQA context mapping
            context_titles = item['context']['title']
            context_sentences = item['context']['sentences']
            
            title_to_text = {}
            for title, sentences in zip(context_titles, context_sentences):
                title_to_text[title] = "".join(sentences)
            
            supp_titles = set(item['supporting_facts']['title'])
            documents = []
            for title in supp_titles:
                if title in title_to_text:
                    documents.append(title_to_text[title])
            
            examples.append(QAExample(
                qid=item['id'],
                dataset_name="hotpotqa",
                question=item['question'],
                answers=[item['answer']],
                documents=documents,
                metadata={"type": item['type'], "level": item['level']}
            ))
            
        return cls(examples)

class NQOpenDataset(BaseQADataset):
    @classmethod
    def from_hf(cls, split: str = "train", max_samples: Optional[int] = None) -> 'NQOpenDataset':
        print(f"Loading NQ Open split={split}...")
        try:
            ds = hf_datasets.load_dataset("nq_open", split=split)
        except Exception as e:
            print(f"Error loading NQ Open: {e}")
            return cls([])

        if max_samples:
            ds = ds.select(range(min(len(ds), max_samples)))

        examples = []
        for i, item in enumerate(ds):
            examples.append(QAExample(
                qid=str(i),
                dataset_name="nq_open",
                question=item['question'],
                answers=item['answer'],
                documents=[],
                metadata={}
            ))
            
        return cls(examples)

class RAGIGBenchDataset(BaseQADataset):
    @classmethod
    def from_hf(cls, split: str = "train", max_samples: Optional[int] = None) -> 'RAGIGBenchDataset':
        print(f"Loading RAG-IGBench split={split}...")
        try:
            ds = hf_datasets.load_dataset("Muyi13/RAG-IGBench", split=split)
        except Exception as e:
            print(f"Error loading RAG-IGBench: {e}")
            return cls([])

        if max_samples:
            ds = ds.select(range(min(len(ds), max_samples)))

        examples = []
        for item in ds:
            examples.append(QAExample(
                qid=str(item['id']),
                dataset_name="rag_igbench",
                question=item['query'],
                answers=[item['gt_clean_answer']],
                documents=item['documents'],
                images=item['images'],
                metadata={"category": item.get('category'), "split": split}
            ))
            
        return cls(examples)

class MMTabDataset(BaseQADataset):
    """
    Adapter for MMTab dataset (Multimodal Table QA).
    """
    @classmethod
    def from_hf(cls, split: str = "train", max_samples: Optional[int] = None) -> 'MMTabDataset':
        print(f"Loading MMTab split={split}...")
        try:
            ds = hf_datasets.load_dataset("SpursgoZmy/MMTab", split=split)
        except Exception as e:
            print(f"Warning: MMTab load failed. Error: {e}")
            return cls([])
            
        if max_samples:
            ds = ds.select(range(min(len(ds), max_samples)))

        examples = []
        for i, item in enumerate(ds):
            question = item.get('question', '')
            answer = item.get('answer', '')
            table = item.get('table', {}) 
            table_str = str(table)
            
            ans_list = [answer] if isinstance(answer, str) else answer
            
            examples.append(QAExample(
                qid=str(item.get('id', i)),
                dataset_name="mmtab",
                question=question,
                answers=ans_list,
                documents=[table_str],
                metadata={"origin": "mmtab"}
            ))
        return cls(examples)

class ComTQADataset(BaseQADataset):
    """
    Adapter for ByteDance/ComTQA dataset.
    """
    @classmethod
    def from_hf(cls, split: str = "train", max_samples: Optional[int] = None) -> 'ComTQADataset':
        print(f"Loading ComTQA split={split}...")
        try:
            ds = hf_datasets.load_dataset("ByteDance/ComTQA", split=split)
        except Exception as e:
            print(f"Warning: ComTQA load failed. Error: {e}")
            return cls([])
            
        if max_samples:
            ds = ds.select(range(min(len(ds), max_samples)))

        examples = []
        for i, item in enumerate(ds):
            # ComTQA typical structure: 'question', 'answer', 'image' (PIL)
            question = item.get('question', '')
            answer = item.get('answer', '')
            
            # Handle image if present (we store path or placeholder or serialized info)
            # For now, we note it in metadata
            image_info = "Available" if item.get('image') else "None"
            
            ans_list = [answer] if isinstance(answer, str) else answer
            
            examples.append(QAExample(
                qid=str(item.get('id', i)),
                dataset_name="comtqa",
                question=question,
                answers=ans_list,
                documents=["[Image Table Context]"], # Placeholder for pure visual tables
                metadata={"origin": "comtqa", "has_image": image_info}
            ))
        return cls(examples)

class RealHiTBenchDataset(BaseQADataset):
    """
    Adapter for RealHiTBench (Hierarchical Tables).
    Loads from local cloned repository if available.
    """
    @classmethod
    def from_local(cls, data_path: str = "RealHiTBench", max_samples: Optional[int] = None) -> 'RealHiTBenchDataset':
        print(f"Loading RealHiTBench from {data_path}...")
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            print("BeautifulSoup not found. Please install beautifulsoup4.")
            return cls([])

        # QA paths
        qa_file = os.path.join(data_path, "QA_final.json")
        html_dir = os.path.join(data_path, "html")
        
        if not os.path.exists(qa_file):
            print(f"Warning: QA file {qa_file} not found.")
            return cls([])
            
        with open(qa_file, 'r') as f:
            data = json.load(f)
            # Schema: {"queries": [...]}
            queries = data.get("queries", [])
            
        print(f"Found {len(queries)} queries in QA_final.json.")
        
        examples = []
        count = 0
        
        for item in queries:
            if max_samples and count >= max_samples:
                break
                
            fname = item.get("FileName")
            # HTML file likely {fname}.html
            html_path = os.path.join(html_dir, f"{fname}.html")
            
            structure_data = {"cells": [], "text": ""}
            full_text = []

            # Parse HTML
            if os.path.exists(html_path):
                try:
                    with open(html_path, 'r', encoding='utf-8') as hf:
                        soup = BeautifulSoup(hf, 'html.parser')
                        
                    # Extract structure
                    # Simple TABLE parsing implementation
                    table = soup.find('table')
                    if table:
                        rows = table.find_all('tr')
                        for r_idx, row in enumerate(rows):
                            cols = row.find_all(['td', 'th'])
                            c_idx = 0
                            for col in cols:
                                cell_text = col.get_text(strip=True)
                                colspan = int(col.get('colspan', 1))
                                rowspan = int(col.get('rowspan', 1))
                                
                                structure_data["cells"].append({
                                    "text": cell_text,
                                    "row": r_idx,
                                    "col": c_idx,
                                    "rowspan": rowspan,
                                    "colspan": colspan,
                                    "is_header": col.name == 'th'
                                })
                                full_text.append(cell_text)
                                c_idx += colspan
                                
                    structure_data["text"] = " ".join(full_text)
                except Exception as e:
                    print(f"Error parsing {html_path}: {e}")
                    structure_data["text"] = "Error parsing table"
            else:
                # Fallback if no HTML (e.g. only excel provided in some versions?)
                # Try simple flattened fallback
                structure_data["text"] = "Table file not found"

            ex = QAExample(
                qid=str(item.get('id', count)),
                dataset_name="realhitbench",
                question=item.get('Question', ''),
                answers=[str(item.get('Answer', ''))], # 'Answer' usually present
                documents=[structure_data["text"]],
                metadata={
                    "file": fname,
                    "structure": structure_data, # Explicit Key for Distiller
                    "source": "realhitbench_html"
                }
            )
            examples.append(ex)
            count += 1
                
        print(f"Loaded {len(examples)} samples from RealHiTBench.")
        return cls(examples)
