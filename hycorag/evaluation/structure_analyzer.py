"""
Structure analyzer for extracting header paths and validating structural concepts.
"""
from typing import Dict, List, Tuple, Optional
from bs4 import BeautifulSoup
import os

def extract_header_hierarchy(html_path: str) -> Dict[Tuple[int, int], List[str]]:
    """
    Extract header paths for each data cell in a table.
    Uses heuristics for tables without explicit <th> tags.
    
    Args:
        html_path: Path to HTML file containing table
        
    Returns:
        Dictionary mapping cell coordinates to header paths
        Format: {(row, col): [header1, header2, ...]}
    """
    if not os.path.exists(html_path):
        return {}
        
    with open(html_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')
    
    table = soup.find('table')
    if not table:
        return {}
    
    # Parse table structure
    rows = table.find_all('tr')
    
    # Step 1: Identify header rows using multiple heuristics
    header_rows = []
    data_start_row = 0
    
    for r_idx, row in enumerate(rows):
        cells = row.find_all(['td', 'th'])
        
        # Heuristic 1: <th> tags
        th_count = len(row.find_all('th'))
        if th_count > len(cells) / 2:
            header_rows.append(r_idx)
            continue
        
        # Heuristic 2: First N rows (assume first 2 rows are headers if no <th>)
        if r_idx < 2 and not header_rows:
            # Check if cells look like headers (short text, no numbers)
            is_header_like = True
            for cell in cells:
                text = cell.get_text(strip=True)
                # If cell has many digits, likely data not header
                if len(text) > 50 or (text and sum(c.isdigit() for c in text) / len(text) > 0.3):
                    is_header_like = False
                    break
            
            if is_header_like:
                header_rows.append(r_idx)
            else:
                data_start_row = r_idx
                break
        else:
            if not data_start_row:
                data_start_row = r_idx
            break
    
    # If no headers found, assume first row
    if not header_rows and len(rows) > 0:
        header_rows = [0]
        data_start_row = 1
    
    # Step 2: Build header hierarchy
    # For each column, track the header path from top to bottom
    col_headers = {}  # {col_idx: [header_text_list]}
    
    for h_row_idx in header_rows:
        row = rows[h_row_idx]
        cells = row.find_all(['td', 'th'])
        
        col_offset = 0
        for cell in cells:
            text = cell.get_text(strip=True)
            colspan = int(cell.get('colspan', 1))
            
            # Add this header to all columns it spans
            for c in range(col_offset, col_offset + colspan):
                if c not in col_headers:
                    col_headers[c] = []
                col_headers[c].append(text)
            
            col_offset += colspan
    
    # Step 3: Map data cells to their header paths
    cell_to_headers = {}
    
    for r_idx in range(data_start_row, len(rows)):
        row = rows[r_idx]
        cells = row.find_all(['td', 'th'])
        
        col_offset = 0
        for cell in cells:
            colspan = int(cell.get('colspan', 1))
            
            # Get header path for this column
            header_path = col_headers.get(col_offset, [])
            cell_to_headers[(r_idx, col_offset)] = header_path
            
            col_offset += colspan
    
    return cell_to_headers

def analyze_structural_coverage(
    predicted_text: str,
    gold_header_paths: Dict[Tuple[int, int], List[str]],
    table_cells: List[Dict]
) -> Dict[str, float]:
    """
    Analyze how well the predicted answer covers structural elements.
    
    Args:
        predicted_text: Generated answer
        gold_header_paths: Ground truth header paths
        table_cells: List of cell dicts with 'text', 'row', 'col'
        
    Returns:
        Metrics dict with header_coverage, cell_reference_accuracy, etc.
    """
    metrics = {
        "header_mention_rate": 0.0,
        "cell_value_accuracy": 0.0,
        "structural_coherence": 0.0
    }
    
    if not gold_header_paths or not table_cells:
        return metrics
    
    # Check header mentions
    all_headers = set()
    for headers in gold_header_paths.values():
        all_headers.update(headers)
    
    mentioned_headers = sum(1 for h in all_headers if h.lower() in predicted_text.lower())
    metrics["header_mention_rate"] = mentioned_headers / max(1, len(all_headers))
    
    # Check cell value mentions
    cell_values = [c['text'] for c in table_cells if c['text']]
    mentioned_values = sum(1 for v in cell_values if v.lower() in predicted_text.lower())
    metrics["cell_value_accuracy"] = mentioned_values / max(1, len(cell_values))
    
    return metrics

def compare_structural_errors(
    baseline_answers: List[str],
    hycorag_answers: List[str],
    ground_truth_paths: List[Dict[Tuple[int, int], List[str]]],
    table_cells_list: List[List[Dict]]
) -> Dict[str, Dict[str, float]]:
    """
    Compare structural errors between Baseline and HyCoRAG.
    
    Returns:
        {
            "baseline": {metrics},
            "hycorag": {metrics},
            "improvement": {metrics}
        }
    """
    baseline_metrics = []
    hycorag_metrics = []
    
    for i in range(len(baseline_answers)):
        b_metrics = analyze_structural_coverage(
            baseline_answers[i],
            ground_truth_paths[i],
            table_cells_list[i]
        )
        h_metrics = analyze_structural_coverage(
            hycorag_answers[i],
            ground_truth_paths[i],
            table_cells_list[i]
        )
        
        baseline_metrics.append(b_metrics)
        hycorag_metrics.append(h_metrics)
    
    # Aggregate
    def avg_metrics(metrics_list):
        if not metrics_list:
            return {}
        keys = metrics_list[0].keys()
        return {k: sum(m[k] for m in metrics_list) / len(metrics_list) for k in keys}
    
    baseline_avg = avg_metrics(baseline_metrics)
    hycorag_avg = avg_metrics(hycorag_metrics)
    
    improvement = {
        k: hycorag_avg[k] - baseline_avg[k] 
        for k in baseline_avg.keys()
    }
    
    return {
        "baseline": baseline_avg,
        "hycorag": hycorag_avg,
        "improvement": improvement
    }
