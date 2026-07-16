"""
Fix the 9x9 fine-tuning notebook with the same bug fixes that solved the 6x6 CUDA crash.

Root cause: During evaluation, the last test batch gets padded with blank_identifier_id=1,
but CastedSparseEmbedding only has num_puzzle_identifiers entries (could be 1).
Index 1 is out of bounds, triggering a CUDA device-side assert.

This script applies the fix to kaggle_finetune_9x9.ipynb.
"""
import json

with open('kaggle_finetune_9x9.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    source = ''.join(cell['source'])
    
    # Fix 1: num_puzzle_identifiers OOB bug in create_model
    if 'num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,' in source:
        cell['source'] = [
            line.replace(
                'num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,',
                'num_puzzle_identifiers=max(train_metadata.num_puzzle_identifiers, train_metadata.blank_identifier_id + 1),'
            ) for line in cell['source']
        ]
        print("Fixed: num_puzzle_identifiers OOB bug")

with open('kaggle_finetune_9x9.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Done! kaggle_finetune_9x9.ipynb has been patched.")
