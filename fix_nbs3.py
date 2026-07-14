import json
with open('kaggle_finetune_6x6.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if i == 6:
        # Restore the dataset generation cell
        cell['source'] = [
            '!python dataset/build_6x6_sudoku_dataset.py --output-dir data/sudoku-6x6 --num-train 1000 --num-test 100 --holes 20\n',
            '\n',
            'import os\n',
            'print("\\nDataset created:")\n',
            'for split in ["train", "test"]:\n',
            '    split_dir = f"data/sudoku-6x6/{split}"\n',
            '    if os.path.exists(split_dir):\n',
            '        files = os.listdir(split_dir)\n',
            '        print(f"  {split}: {files}\\n")\n'
        ]

with open('kaggle_finetune_6x6.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
