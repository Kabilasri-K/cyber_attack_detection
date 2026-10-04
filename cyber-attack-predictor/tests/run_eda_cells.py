"""
run_eda_cells.py — extracts and executes all code cells from 01_eda.ipynb
as a single script so we can validate correctness without a Jupyter kernel.
Uses matplotlib's non-interactive Agg backend (no display window needed).
"""
import json
import sys
import os
from pathlib import Path

# Use non-interactive backend so matplotlib doesn't try to open a window
import matplotlib
matplotlib.use('Agg')

# The notebook uses Path().resolve() expecting to be IN the notebooks/ folder.
# We chdir into notebooks/ before executing, so paths resolve correctly.
project_root = Path(__file__).resolve().parent.parent
os.chdir(project_root / 'notebooks')

# Load the notebook (path relative to project root)
with open(project_root / 'notebooks' / '01_eda.ipynb') as f:
    nb = json.load(f)

code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
print(f"Executing {len(code_cells)} code cells...\n")

# Build a shared namespace across all cells (simulates Jupyter kernel state)
namespace = {}

for i, cell in enumerate(code_cells, 1):
    source = ''.join(cell['source'])
    cell_id = cell.get('id', f'cell-{i}')
    print(f"{'='*60}")
    print(f"Cell {i}: {cell_id}")
    print(f"{'='*60}")
    try:
        exec(compile(source, f'<cell-{i}>', 'exec'), namespace)
        print(f"  [OK]\n")
    except Exception as exc:
        print(f"  [ERROR] {type(exc).__name__}: {exc}\n", file=sys.stderr)
        sys.exit(1)

print("\nAll cells executed successfully.")
