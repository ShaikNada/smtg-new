import json

with open('anly1.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    print(f'=== Cell {i} [{cell["cell_type"]}] ===')
    print(src)
    print()
