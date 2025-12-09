"""
Test script for structure analyzer on RealHiTBench.
"""
import sys
sys.path.insert(0, '/home/user/HyCoRAG')

from hycorag.evaluation.structure_analyzer import extract_header_hierarchy
import os

# Test on first RealHiTBench sample
html_path = "RealHiTBench/html/employment-table01.html"

if not os.path.exists(html_path):
    print(f"File not found: {html_path}")
    sys.exit(1)

print(f"Analyzing: {html_path}")
print("="*60)

header_paths = extract_header_hierarchy(html_path)

print(f"\nExtracted {len(header_paths)} cell-to-header mappings")
print("\nSample mappings (first 10):")

for i, ((row, col), headers) in enumerate(list(header_paths.items())[:10]):
    print(f"  Cell ({row}, {col}): {' → '.join(headers)}")

# Analyze header hierarchy depth
depths = [len(headers) for headers in header_paths.values()]
if depths:
    print(f"\nHeader hierarchy statistics:")
    print(f"  Min depth: {min(depths)}")
    print(f"  Max depth: {max(depths)}")
    print(f"  Avg depth: {sum(depths)/len(depths):.2f}")

print("\n" + "="*60)
print("Structure analysis complete!")
