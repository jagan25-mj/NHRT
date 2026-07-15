import json

with open("finetune_lora.py", "r", encoding="utf-8") as f:
    script_content = f.read()

# Revert cpu back to cuda for Kaggle T4 GPUs
script_content = script_content.replace('"cpu"', '"cuda"').replace('.cpu()', '.cuda()').replace('.cuda().numpy()', '.cpu().numpy()')
script_lines = [line + "\n" for line in script_content.split("\n")]

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# HRM LoRA Fine-Tuning on Kaggle (T4 GPU)\n",
                "This notebook sets up the environment, downloads the pre-trained HRM checkpoint, builds the Sudoku Extreme dataset, and runs a custom parameter-efficient LoRA fine-tuning script on a T4 GPU."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!git clone https://github.com/sapientinc/HRM.git\n",
                "%cd HRM\n",
                "!sed -i '/adam-atan2/d' requirements.txt\n",
                "!pip install -r requirements.txt peft huggingface_hub wandb"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Patch models/layers.py to bypass flash_attn requirement and use PyTorch native SDPA\n",
                "patch_code = '''\\\n",
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
                "    f.write(content)\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Build the Sudoku Extreme Dataset\n",
                "!sed -i 's/if set_name == \"train\" and config.subsample_size is not None:/if config.subsample_size is not None:/g' dataset/build_sudoku_dataset.py\n",
                "!python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 2. Download the pre-trained HRM checkpoint\n",
                "import os\n",
                "from huggingface_hub import snapshot_download\n",
                "os.makedirs('checkpoints/HRM-checkpoint-sudoku-extreme', exist_ok=True)\n",
                "snapshot_download(repo_id='sapientinc/HRM-checkpoint-sudoku-extreme', local_dir='checkpoints/HRM-checkpoint-sudoku-extreme')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["%%writefile finetune_lora.py\n"] + script_lines
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 4. Run the Fine-Tuning on T4 GPU\n",
                "!WANDB_MODE=disabled python finetune_lora.py data_path=data/sudoku-extreme-1k-aug-1000 epochs=2 eval_interval=1 lr=1e-5 puzzle_emb_lr=1e-5 weight_decay=1.0 puzzle_emb_weight_decay=1.0 global_batch_size=8 +load_checkpoint=checkpoints/HRM-checkpoint-sudoku-extreme/checkpoint"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Expected Results\n",
                "- **Trainable Parameters**: ~558,082 (only ~2% of the base 27M parameters are trainable due to LoRA injection).\n",
                "- **VRAM Usage**: Fits comfortably within the 16GB VRAM of a Kaggle T4 GPU thanks to frozen base weights and a batch size of 8.\n",
                "- **Performance**: The script uses `torch.compile` and mixed precision on the T4 GPU, leading to fast iteration times.\n",
                "- **Outputs**: Checkpoints will be saved in the `checkpoints/` directory at the end of each evaluation epoch, with the ACT (Adaptive Computation Time) halting logic dynamically fine-tuned for the augmented dataset."
            ]
        }
    ],
    "metadata": {
        "accelerator": "GPU",
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("kaggle_finetune_lora.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Created kaggle_finetune_lora.ipynb")
