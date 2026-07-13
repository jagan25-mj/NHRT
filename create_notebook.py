import base64
import json

with open("hrm_code.zip", "rb") as f:
    zip_data = f.read()

b64_zip = base64.b64encode(zip_data).decode('utf-8')

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Hierarchical Reasoning Model - Kaggle Runner\n",
                "This notebook contains the entire HRM codebase. Run the cells sequentially to extract the code, install dependencies, and run the training on a Kaggle GPU."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import base64\n",
                "import zipfile\n",
                "import os\n",
                "\n",
                "b64_data = \"" + b64_zip + "\"\n",
                "zip_bytes = base64.b64decode(b64_data)\n",
                "\n",
                "with open('hrm_code.zip', 'wb') as f:\n",
                "    f.write(zip_bytes)\n",
                "\n",
                "print('Extracting codebase...')\n",
                "with zipfile.ZipFile('hrm_code.zip', 'r') as zip_ref:\n",
                "    zip_ref.extractall('hrm_project')\n",
                "print('Extraction complete. Files are in the hrm_project directory.')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "%cd /kaggle/working/hrm_project\n",
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
                "# Disable wandb or login to wandb before running\n",
                "import os\n",
                "os.environ['WANDB_MODE'] = 'offline'\n",
                "\n",
                "!python pretrain.py data_path=data/sudoku-extreme-1k-aug-1000 epochs=5 eval_interval=1 global_batch_size=128 lr=7e-5 puzzle_emb_lr=7e-5 weight_decay=1.0 puzzle_emb_weight_decay=1.0\n"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("kaggle_hrm_runner.ipynb", "w", encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print("Created kaggle_hrm_runner.ipynb")
