import collections
import string
import re
from typing import List

def normalize_answer(s):
    """
    Improved normalization for table QA answers.
    Handles numbers, units, percentages, and common variations.
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()
    
    def normalize_numbers(text):
        # Normalize number formats: 1,000 -> 1000, $100M -> 100M
        text = re.sub(r'[\$€£¥]', '', text)  # Remove currency symbols
        text = re.sub(r',(\d{3})', r'\1', text)  # Remove thousand separators
        return text

    # Apply all normalizations
    s = str(s)
    s = normalize_numbers(s)
    s = lower(s)
    s = remove_punc(s)
    s = remove_articles(s)
    s = white_space_fix(s)
    
    return s

def calculate_em(prediction, ground_truth):
    """Exact match score."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))

def calculate_f1(prediction, ground_truth):
    """F1 score."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def calculate_hit_at_k(retrieved_ids, gold_id, k=None):
    """Hit@K score."""
    return float(gold_id in retrieved_ids)

def calculate_header_path_match(prediction: str, gold_path: List[str]) -> float:
    """
    RQ2 Metric: Checks if the predicted answer implies knowledge of the correct header path.
    Simple heuristic: Check if keywords from gold path appear in prediction or context.
    Real impl would need struct-aware generation.
    """
    if not gold_path: return 0.0
    
    hits = 0
    norm_pred = normalize_answer(prediction)
    for header in gold_path:
        if normalize_answer(header) in norm_pred:
            hits += 1
            
    return hits / len(gold_path)

def calculate_unit_match(prediction: str, gold_answer: str) -> float:
    """
    RQ2 Metric: Checks if units ($, %, kg, etc) match.
    """
    # Simple regex for common units
    unit_pattern = r'([$€£%]|kg|m|cm|mm|s|sec|min|hr)'
    
    pred_units = re.findall(unit_pattern, prediction)
    gold_units = re.findall(unit_pattern, gold_answer)
    
    if not gold_units: return 1.0 # No unit in gold, strict match not required
    if not pred_units: return 0.0 # Unit in gold but not in pred
    
    # Check intersection
    common = set(pred_units) & set(gold_units)
    return float(len(common) > 0)
