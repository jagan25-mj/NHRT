import json
import os

with open('kaggle_evaluate_6x6.ipynb', 'r') as f:
    nb = json.load(f)

# Cell 0: add pip install
nb['cells'][0]['source'] = [
    "!git clone https://github.com/jagan25-mj/NHRT.git\n",
    "%cd NHRT\n",
    "!pip install -r requirements.txt"
]

# Cell 2: add debug info
new_source = [
    "import os\n",
    "import glob\n",
    "\n",
    "# Let's see what is inside /kaggle/input/ to debug!\n",
    "print('Contents of /kaggle/input:')\n",
    "os.system('find /kaggle/input -maxdepth 5')\n",
    "\n",
    "checkpoint_files = glob.glob('/kaggle/input/**/step_*', recursive=True)\n",
    "checkpoint_files = [f for f in checkpoint_files if 'all_preds' not in f]\n",
    "\n",
    "if not checkpoint_files:\n",
    "    print('\\nNo checkpoints found! Make sure you attached the output of the previous notebook.')\n",
    "    print('If it was an interactive session, you might need to run the training in a \"Save Version -> Run All\" commit to save the outputs permanently.')\n",
    "else:\n",
    "    def extract_step(f):\n",
    "        basename = os.path.basename(f)\n",
    "        return int(basename.replace('step_', ''))\n",
    "    \n",
    "    latest_checkpoint = max(checkpoint_files, key=extract_step)\n",
    "    print(f'\\nFound checkpoint: {latest_checkpoint}')\n",
    "    \n",
    "    cmd = f'python evaluate.py checkpoint=\"{latest_checkpoint}\"'\n",
    "    print(f'Running: {cmd}')\n",
    "    os.system(cmd)\n"
]
nb['cells'][2]['source'] = new_source

with open('kaggle_evaluate_6x6.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
