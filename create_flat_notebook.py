import os
import json

def get_all_files():
    files_to_include = []
    for root, dirs, files in os.walk('.'):
        if 'venv' in root or '.git' in root or '__pycache__' in root or '.vscode' in root:
            continue
        for file in files:
            if file.endswith('.py') or file.endswith('.yaml') or file.endswith('.txt'):
                if file in ['create_notebook.py', 'create_flat_notebook.py']:
                    continue
                files_to_include.append(os.path.join(root, file).replace('\\', '/').removeprefix('./'))
    return files_to_include

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Hierarchical Reasoning Model - Flat Kaggle Notebook\n",
                "Run all cells sequentially. The first few cells will recreate the codebase structure on Kaggle using `%%writefile`, and the final cells will run the training."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!mkdir -p dataset/raw-data models/hrm utils config/arch\n"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

for filepath in get_all_files():
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    cell_source = f"%%writefile {filepath}\n" + content
    
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + '\n' for line in cell_source.split('\n')]
    })

# Add the execution cells
notebook["cells"].extend([
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "!pip install -r requirements.txt\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "!python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "os.environ['WANDB_MODE'] = 'offline'\n",
            "!python pretrain.py data_path=data/sudoku-extreme-1k-aug-1000 epochs=5 eval_interval=1 global_batch_size=128 lr=7e-5 puzzle_emb_lr=7e-5 weight_decay=1.0 puzzle_emb_weight_decay=1.0\n"
        ]
    }
])

with open("kaggle_hrm_flat.ipynb", "w", encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print("Created kaggle_hrm_flat.ipynb")
