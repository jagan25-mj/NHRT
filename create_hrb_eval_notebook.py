"""
Creates a Kaggle evaluation notebook for the HRB model.

This notebook:
1. Clones the NHRT repo and installs dependencies
2. Patches flash_attn for T4 GPU compatibility
3. Builds the 6x6 sudoku test dataset
4. Finds the latest HRB checkpoint from attached training output
5. Runs evaluation and reports cell/puzzle accuracy

Usage on Kaggle:
- Create a new notebook
- Attach the output of the "HRB-FINE" training notebook as a data source
- Upload and run this notebook
"""
import json

notebook = {
    "cells": [
        # ---- Cell 0: Title ----
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# HRB Model Evaluation (6×6 Sudoku)\n",
                "\n",
                "This notebook evaluates the partially-trained HRB (Hierarchical Reasoning Block) model on 6×6 Sudoku puzzles.\n",
                "\n",
                "**Prerequisites**: Attach the output of the `HRB-FINE` training notebook as a data source.\n",
                "The checkpoint files will be found automatically in `/kaggle/input/`."
            ]
        },
        # ---- Cell 1: Clone repo + install deps ----
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!git clone https://github.com/jagan25-mj/NHRT.git\n",
                "%cd NHRT\n",
                "!pip install -r requirements.txt peft huggingface_hub"
            ]
        },
        # ---- Cell 2: Patch flash_attn ----
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Patch models/layers.py to bypass flash_attn and use PyTorch native SDPA\n",
                "import torch.nn.functional as F\n",
                "patch_code = '''\n",
                "\n",
                "import torch.nn.functional as F\n",
                "def flash_attn_func(q, k, v, causal=False):\n",
                "    q = q.transpose(1, 2)\n",
                "    k = k.transpose(1, 2)\n",
                "    v = v.transpose(1, 2)\n",
                "    out = F.scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous(), is_causal=causal)\n",
                "    return out.transpose(1, 2).contiguous()\n",
                "'''\n",
                "with open('models/layers.py', 'r') as f:\n",
                "    content = f.read()\n",
                "content = content.replace('from flash_attn_interface import flash_attn_func', 'pass')\n",
                "content = content.replace('from flash_attn import flash_attn_func', 'pass')\n",
                "content += patch_code\n",
                "with open('models/layers.py', 'w') as f:\n",
                "    f.write(content)\n",
                "print('Patched models/layers.py')"
            ]
        },
        # ---- Cell 3: Build 6x6 sudoku dataset ----
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Build the 6x6 Sudoku dataset (same as used during training)\n",
                "!python dataset/build_6x6_sudoku_dataset.py\n",
                "print('\\n6x6 Sudoku dataset built successfully.')"
            ]
        },
        # ---- Cell 4: Find checkpoint + debug ----
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import glob\n",
                "\n",
                "# Debug: show what's in /kaggle/input/\n",
                "print('=== Contents of /kaggle/input/ ===')\n",
                "os.system('find /kaggle/input -maxdepth 5 -type f | head -50')\n",
                "print()\n",
                "\n",
                "# Find all checkpoint files (step_XXXXX format)\n",
                "checkpoint_files = glob.glob('/kaggle/input/**/step_*', recursive=True)\n",
                "checkpoint_files = [f for f in checkpoint_files if 'all_preds' not in f]\n",
                "\n",
                "if not checkpoint_files:\n",
                "    print('\\n❌ No checkpoints found!')\n",
                "    print('Make sure you attached the output of the HRB-FINE training notebook.')\n",
                "    print('If you ran it as an interactive session, you need to do \"Save Version -> Run All\" first.')\n",
                "else:\n",
                "    def extract_step(f):\n",
                "        basename = os.path.basename(f)\n",
                "        return int(basename.replace('step_', ''))\n",
                "    \n",
                "    checkpoint_files.sort(key=extract_step)\n",
                "    print(f'\\n✅ Found {len(checkpoint_files)} checkpoint(s):')\n",
                "    for cf in checkpoint_files:\n",
                "        print(f'  Step {extract_step(cf):>6d}: {cf}')\n",
                "    \n",
                "    latest_checkpoint = checkpoint_files[-1]\n",
                "    latest_step = extract_step(latest_checkpoint)\n",
                "    print(f'\\n📌 Will evaluate: step_{latest_step} ({latest_checkpoint})')"
            ]
        },
        # ---- Cell 5: Copy config + prepare evaluation ----
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os, glob, shutil\n",
                "\n",
                "# Find the config file alongside the checkpoint\n",
                "checkpoint_files = glob.glob('/kaggle/input/**/step_*', recursive=True)\n",
                "checkpoint_files = [f for f in checkpoint_files if 'all_preds' not in f]\n",
                "\n",
                "def extract_step(f):\n",
                "    return int(os.path.basename(f).replace('step_', ''))\n",
                "\n",
                "latest_checkpoint = max(checkpoint_files, key=extract_step)\n",
                "ckpt_dir = os.path.dirname(latest_checkpoint)\n",
                "latest_step = extract_step(latest_checkpoint)\n",
                "\n",
                "# Check if all_config.yaml exists alongside the checkpoint\n",
                "config_path = os.path.join(ckpt_dir, 'all_config.yaml')\n",
                "if os.path.exists(config_path):\n",
                "    print(f'✅ Found config at: {config_path}')\n",
                "else:\n",
                "    print(f'⚠️  No all_config.yaml found at {ckpt_dir}')\n",
                "    print('Will need to reconstruct config manually.')\n",
                "\n",
                "# Copy checkpoint to a writable location\n",
                "work_ckpt_dir = f'/kaggle/working/eval_checkpoint'\n",
                "os.makedirs(work_ckpt_dir, exist_ok=True)\n",
                "shutil.copy2(latest_checkpoint, os.path.join(work_ckpt_dir, f'step_{latest_step}'))\n",
                "if os.path.exists(config_path):\n",
                "    shutil.copy2(config_path, os.path.join(work_ckpt_dir, 'all_config.yaml'))\n",
                "\n",
                "# Also copy any other files from the checkpoint directory\n",
                "for f in os.listdir(ckpt_dir):\n",
                "    src = os.path.join(ckpt_dir, f)\n",
                "    dst = os.path.join(work_ckpt_dir, f)\n",
                "    if not os.path.exists(dst) and os.path.isfile(src):\n",
                "        shutil.copy2(src, dst)\n",
                "\n",
                "print(f'\\nCopied checkpoint to: {work_ckpt_dir}')\n",
                "print('Contents:')\n",
                "for f in sorted(os.listdir(work_ckpt_dir)):\n",
                "    size_mb = os.path.getsize(os.path.join(work_ckpt_dir, f)) / 1024 / 1024\n",
                "    print(f'  {f} ({size_mb:.1f} MB)')"
            ]
        },
        # ---- Cell 6: Run evaluation ----
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os, yaml, glob, torch\n",
                "import numpy as np\n",
                "\n",
                "os.environ['DISABLE_COMPILE'] = '1'\n",
                "\n",
                "from finetune_lora import PretrainConfig, init_train_state, create_dataloader, evaluate\n",
                "from models.losses import IGNORE_LABEL_ID\n",
                "\n",
                "# Find checkpoint\n",
                "work_ckpt_dir = '/kaggle/working/eval_checkpoint'\n",
                "ckpt_files = [f for f in os.listdir(work_ckpt_dir) if f.startswith('step_') and 'all_preds' not in f]\n",
                "ckpt_files.sort(key=lambda f: int(f.replace('step_', '')))\n",
                "latest_ckpt = ckpt_files[-1]\n",
                "latest_step = int(latest_ckpt.replace('step_', ''))\n",
                "ckpt_path = os.path.join(work_ckpt_dir, latest_ckpt)\n",
                "\n",
                "print(f'Evaluating checkpoint: {latest_ckpt} (step {latest_step})')\n",
                "print('=' * 60)\n",
                "\n",
                "# Load config\n",
                "config_path = os.path.join(work_ckpt_dir, 'all_config.yaml')\n",
                "with open(config_path, 'r') as f:\n",
                "    config = PretrainConfig(**yaml.safe_load(f))\n",
                "\n",
                "config.eval_save_outputs = ['logits']\n",
                "config.checkpoint_path = work_ckpt_dir\n",
                "config.load_checkpoint = None\n",
                "\n",
                "# Override data_path to the 6x6 sudoku dataset we just built\n",
                "config.data_path = 'data/sudoku-6x6'\n",
                "\n",
                "print(f'Architecture: {config.arch.name}')\n",
                "print(f'Data path: {config.data_path}')\n",
                "print(f'Batch size: {config.global_batch_size}')\n",
                "\n",
                "# Create test dataloader\n",
                "eval_loader, eval_metadata = create_dataloader(\n",
                "    config, 'test', test_set_mode=True,\n",
                "    epochs_per_iter=1, global_batch_size=config.global_batch_size,\n",
                "    rank=0, world_size=1\n",
                ")\n",
                "\n",
                "# Also create train loader for metadata (needed by init_train_state)\n",
                "train_loader, train_metadata = create_dataloader(\n",
                "    config, 'train', test_set_mode=False,\n",
                "    epochs_per_iter=1, global_batch_size=config.global_batch_size,\n",
                "    rank=0, world_size=1\n",
                ")\n",
                "\n",
                "# Init model\n",
                "train_state = init_train_state(config, train_metadata, world_size=1)\n",
                "\n",
                "# Load checkpoint weights\n",
                "print(f'\\nLoading weights from {ckpt_path}...')\n",
                "state_dict = torch.load(ckpt_path, map_location='cuda', weights_only=True)\n",
                "try:\n",
                "    train_state.model.load_state_dict(state_dict, assign=True)\n",
                "except:\n",
                "    clean_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}\n",
                "    train_state.model.load_state_dict(clean_dict, assign=True, strict=False)\n",
                "print('Weights loaded.')\n",
                "\n",
                "train_state.step = latest_step\n",
                "train_state.model.eval()\n",
                "print(f'\\nRunning evaluation...')\n",
                "\n",
                "# Use the built-in evaluate function\n",
                "metrics = evaluate(config, train_state, eval_loader, eval_metadata, rank=0, world_size=1)\n",
                "\n",
                "if metrics is not None:\n",
                "    print('\\n' + '=' * 60)\n",
                "    print(f'EVALUATION RESULTS (Step {latest_step})')\n",
                "    print('=' * 60)\n",
                "    for set_name, set_metrics in metrics.items():\n",
                "        print(f'\\n  Set: {set_name}')\n",
                "        for metric_name, value in set_metrics.items():\n",
                "            print(f'    {metric_name}: {value:.6f}')\n",
                "else:\n",
                "    print('\\n⚠️  No metrics returned. Check the evaluation output above.')"
            ]
        },
        # ---- Cell 7: Manual cell-level accuracy check ----
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Additional manual accuracy check with detailed per-puzzle breakdown\n",
                "import torch\n",
                "from models.losses import IGNORE_LABEL_ID\n",
                "\n",
                "total_correct = 0\n",
                "total_cells = 0\n",
                "total_exact = 0\n",
                "total_puzzles = 0\n",
                "max_iters = 16\n",
                "\n",
                "print('Running detailed per-batch evaluation...')\n",
                "print('=' * 60)\n",
                "\n",
                "train_state.model.eval()\n",
                "batch_id = 0\n",
                "\n",
                "with torch.inference_mode():\n",
                "    for set_name, batch, global_batch_size in eval_loader:\n",
                "        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}\n",
                "        \n",
                "        with torch.device('cuda'):\n",
                "            carry = train_state.model.initial_carry(batch)\n",
                "        \n",
                "        for step_i in range(max_iters):\n",
                "            carry, _, metrics, preds, all_finish = train_state.model(\n",
                "                carry=carry, batch=batch, return_keys=['logits'])\n",
                "            if all_finish:\n",
                "                break\n",
                "        \n",
                "        labels = batch['labels']\n",
                "        logits = preds.get('logits', None)\n",
                "        \n",
                "        if logits is not None:\n",
                "            predictions = torch.argmax(logits, dim=-1)\n",
                "            mask = labels != IGNORE_LABEL_ID\n",
                "            \n",
                "            correct = (predictions == labels) & mask\n",
                "            batch_correct = correct.sum().item()\n",
                "            batch_cells = mask.sum().item()\n",
                "            total_correct += batch_correct\n",
                "            total_cells += batch_cells\n",
                "            \n",
                "            batch_exact = 0\n",
                "            for i in range(labels.shape[0]):\n",
                "                puzzle_mask = mask[i]\n",
                "                if puzzle_mask.sum() > 0:\n",
                "                    total_puzzles += 1\n",
                "                    if correct[i].sum() == puzzle_mask.sum():\n",
                "                        total_exact += 1\n",
                "                        batch_exact += 1\n",
                "            \n",
                "            batch_acc = 100 * batch_correct / max(batch_cells, 1)\n",
                "            print(f'  Batch {batch_id} ({set_name}): '\n",
                "                  f'Cell {batch_correct}/{batch_cells} ({batch_acc:.1f}%) | '\n",
                "                  f'Exact: {batch_exact}/{labels.shape[0]}')\n",
                "        \n",
                "        batch_id += 1\n",
                "\n",
                "print()\n",
                "print('=' * 60)\n",
                "print(f'FINAL RESULTS (Step {latest_step})')\n",
                "print('=' * 60)\n",
                "cell_acc = 100 * total_correct / max(total_cells, 1)\n",
                "puzzle_acc = 100 * total_exact / max(total_puzzles, 1)\n",
                "print(f'  Cell Accuracy:   {total_correct:>6d}/{total_cells:<6d} = {cell_acc:.2f}%')\n",
                "print(f'  Puzzle Accuracy: {total_exact:>6d}/{total_puzzles:<6d} = {puzzle_acc:.2f}%')\n",
                "print(f'  Halting Steps:   {step_i + 1} / {max_iters}')\n",
                "print('=' * 60)"
            ]
        }
    ],
    "metadata": {
        "accelerator": "GPU",
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("kaggle_evaluate_hrb.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Created kaggle_evaluate_hrb.ipynb")
print("\nTo use on Kaggle:")
print("1. Upload kaggle_evaluate_hrb.ipynb to Kaggle as a new notebook")
print("2. Attach the output of your 'HRB-FINE' training notebook as a data source")
print("3. Enable GPU (T4) accelerator")
print("4. Run All cells")
