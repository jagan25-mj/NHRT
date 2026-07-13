import os
import json

def get_all_files():
    files_to_include = []
    for root, dirs, files in os.walk('.'):
        if 'venv' in root or '.git' in root or '__pycache__' in root or '.vscode' in root or 'scratch' in root:
            continue
        for file in files:
            if file.endswith('.py') or file.endswith('.yaml') or file.endswith('.txt'):
                if file in ['create_notebook.py', 'create_flat_notebook.py', 'create_6x6_notebook.py']:
                    continue
                files_to_include.append(os.path.join(root, file).replace('\\', '/').removeprefix('./'))
    return files_to_include

# ============================================================
# Notebook structure
# ============================================================
notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# HRM on 6x6 Sudoku - Train + Evaluate (12-Hour Kaggle Session)\n",
                "\n",
                "This notebook trains the Hierarchical Reasoning Model on 6x6 Sudoku puzzles\n",
                "and evaluates it automatically at the end.\n",
                "\n",
                "**Key fixes applied:**\n",
                "- Checkpoints are saved BEFORE evaluation (prevents data loss if eval crashes)\n",
                "- Evaluation is wrapped in try/except (training continues even if eval fails on T4)\n",
                "- Batch size = 64 (prevents OOM on T4 GPU)\n",
                "\n",
                "**Requirements:** GPU T4 x2 accelerator, 12-hour session."
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

# ---- Write all source files ----
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

# ---- Execution cells ----
notebook["cells"].extend([
    # Install dependencies
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Step 1: Install Dependencies"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "!pip install -r requirements.txt\n"
        ]
    },

    # Generate dataset
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Step 2: Generate 6x6 Sudoku Dataset"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "!python dataset/build_6x6_sudoku_dataset.py --output-dir data/sudoku-6x6 --num-train 1000 --num-test 100 --holes 20\n"
        ]
    },

    # Train the model
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 3: Train HRM on 6x6 Sudoku\n",
            "\n",
            "- Checkpoints are saved BEFORE evaluation at each interval\n",
            "- If evaluation crashes on T4 (bfloat16 issue), training continues\n",
            "- ~11.5 hours training, checkpoints every ~69 minutes"
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
            "\n",
            "!python pretrain.py \\\n",
            "    data_path=data/sudoku-6x6 \\\n",
            "    epochs=5000 \\\n",
            "    eval_interval=500 \\\n",
            "    checkpoint_every_eval=True \\\n",
            "    global_batch_size=64 \\\n",
            "    lr=7e-5 \\\n",
            "    puzzle_emb_lr=7e-5 \\\n",
            "    weight_decay=1.0 \\\n",
            "    puzzle_emb_weight_decay=1.0\n"
        ]
    },

    # Evaluate the model
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 4: Evaluate the Trained Model\n",
            "\n",
            "Finds the latest checkpoint and runs evaluation.\n",
            "Uses DISABLE_COMPILE to avoid T4 bfloat16 torch.compile issues during eval."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "\n",
            "# Find the latest checkpoint\n",
            "latest_step = -1\n",
            "best_checkpoint_path = None\n",
            "\n",
            "for root, dirs, files in os.walk('checkpoints'):\n",
            "    for file in files:\n",
            "        if file.startswith('step_') and 'all_preds' not in file:\n",
            "            try:\n",
            "                step_num = int(file.split('_')[1])\n",
            "                if step_num > latest_step:\n",
            "                    latest_step = step_num\n",
            "                    best_checkpoint_path = os.path.join(root, file)\n",
            "            except ValueError:\n",
            "                pass\n",
            "\n",
            "if best_checkpoint_path is None:\n",
            "    print('ERROR: No checkpoint found!')\n",
            "else:\n",
            "    print(f'Latest checkpoint: {best_checkpoint_path}  (step {latest_step})')\n",
            "    print('Running evaluation (with torch.compile disabled for T4 stability)...')\n",
            "    print('=' * 60)\n",
            "    # DISABLE_COMPILE avoids bfloat16 torch.compile issues on T4\n",
            "    os.environ['DISABLE_COMPILE'] = '1'\n",
            "    !python evaluate.py checkpoint=\"{best_checkpoint_path}\"\n",
            "    print('=' * 60)\n",
            "    print('Evaluation complete!')\n"
        ]
    },

    # List checkpoints
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Step 5: List All Saved Checkpoints"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "\n",
            "print('Saved checkpoints:')\n",
            "print('-' * 40)\n",
            "for root, dirs, files in os.walk('checkpoints'):\n",
            "    for file in sorted(files):\n",
            "        if file.startswith('step_') and 'all_preds' not in file:\n",
            "            filepath = os.path.join(root, file)\n",
            "            size_mb = os.path.getsize(filepath) / (1024 * 1024)\n",
            "            print(f'  {file}  ({size_mb:.1f} MB)')\n"
        ]
    }
])

# ---- Write the notebook ----
with open("kaggle_hrm_6x6.ipynb", "w", encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print("Created kaggle_hrm_6x6.ipynb")
print()
print("Fixes applied:")
print("  1. Checkpoint saved BEFORE evaluation (no more lost weights)")
print("  2. Evaluation wrapped in try/except (training continues if eval crashes)")
print("  3. Final eval uses DISABLE_COMPILE (avoids T4 bfloat16 crash)")
