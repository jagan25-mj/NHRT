import os
import json

def get_all_files():
    files_to_include = []
    for root, dirs, files in os.walk('.'):
        if 'venv' in root or '.git' in root or '__pycache__' in root or '.vscode' in root or 'scratch' in root:
            continue
        for file in files:
            if file.endswith('.py') or file.endswith('.yaml') or file == 'requirements.txt':
                if file in ['create_notebook.py', 'create_flat_notebook.py', 'create_6x6_notebook.py', 'create_nhrt_notebooks.py']:
                    continue
                files_to_include.append(os.path.join(root, file).replace('\\', '/').removeprefix('./'))
    return files_to_include


def generate_notebook(arch_name, nb_filename, title):
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# {title}\n",
                    "\n",
                    f"This notebook fine-tunes the HRM base model using the **{arch_name}** architecture\n",
                    "on 6x6 Sudoku puzzles using LoRA.\n",
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
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Step 1: Install Dependencies & Download Checkpoint"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!pip install -r requirements.txt peft huggingface_hub wandb\n",
                "import os\n",
                "from huggingface_hub import snapshot_download\n",
                "os.makedirs('checkpoints/HRM-checkpoint-sudoku-extreme', exist_ok=True)\n",
                "snapshot_download(repo_id='sapientinc/HRM-checkpoint-sudoku-extreme', local_dir='checkpoints/HRM-checkpoint-sudoku-extreme')\n"
            ]
        },
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
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 3: Fine-Tune HRM on 6x6 Sudoku using LoRA\n",
                "\n",
                f"Using architecture: {arch_name}"
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
                "!python finetune_lora.py \\\n",
                f"    arch={arch_name} \\\n",
                "    data_path=data/sudoku-6x6 \\\n",
                "    epochs=5000 \\\n",
                "    eval_interval=500 \\\n",
                "    checkpoint_every_eval=True \\\n",
                "    global_batch_size=64 \\\n",
                "    lr=7e-5 \\\n",
                "    puzzle_emb_lr=7e-5 \\\n",
                "    weight_decay=1.0 \\\n",
                "    puzzle_emb_weight_decay=1.0 \\\n",
                "    +load_checkpoint=checkpoints/HRM-checkpoint-sudoku-extreme/checkpoint\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 4: Evaluate the Trained Model\n",
                "\n",
                "Finds the latest checkpoint and runs evaluation."
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
                "    os.environ['DISABLE_COMPILE'] = '1'\n",
                "    !python evaluate.py checkpoint=\"{best_checkpoint_path}\"\n"
            ]
        }
    ])

    with open(nb_filename, "w", encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
    print(f"Created {nb_filename}")


if __name__ == '__main__':
    generate_notebook('hrm_hrb_v1', 'kaggle_finetune_hrb_6x6.ipynb', 'HRM Fine-Tuning - HRB Architecture (6x6)')
    generate_notebook('hrm_deep_v1', 'kaggle_finetune_deep_6x6.ipynb', 'HRM Fine-Tuning - DeepReasoner Architecture (6x6)')
    generate_notebook('hrm_v1', 'kaggle_finetune_act_6x6.ipynb', 'HRM Fine-Tuning - AdaptiveDeepReasoner (ACT) (6x6)')
