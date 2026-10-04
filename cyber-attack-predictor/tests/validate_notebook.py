"""Validate that notebooks/01_eda.ipynb is well-formed JSON and summarise its cells."""
import json

with open('notebooks/01_eda.ipynb') as f:
    nb = json.load(f)

cells = nb['cells']
code_cells     = [c for c in cells if c['cell_type'] == 'code']
markdown_cells = [c for c in cells if c['cell_type'] == 'markdown']

print(f'Total cells    : {len(cells)}')
print(f'Code cells     : {len(code_cells)}')
print(f'Markdown cells : {len(markdown_cells)}')
print()
print('Cell inventory:')
for c in cells:
    src_preview = ''.join(c['source'])[:70].replace('\n', ' ')
    print(f"  [{c['cell_type']:8}] {c['id']:30}  {src_preview}...")
print()
print('JSON is valid.')
