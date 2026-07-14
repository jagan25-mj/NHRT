import json

def fix_6x6():
    with open('kaggle_finetune_6x6.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            # Fix pip install
            new_source = []
            for line in cell['source']:
                if '!pip install -r requirements.txt' in line:
                    new_source.append("!pip install setuptools wheel\n")
                    new_source.append("!pip install einops tqdm pydantic wandb omegaconf hydra-core huggingface_hub peft argdantic coolname --quiet\n")
                elif 'from argdantic import ArgParser' in line:
                    new_source.append("import argparse\n")
                elif 'from pydantic import BaseModel' in line:
                    pass
                elif 'cli = ArgParser()' in line:
                    pass
                elif 'class DataProcessConfig' in line:
                    pass
                elif 'output_dir: str' in line or 'num_train: int' in line or 'num_test: int' in line or 'holes: int' in line:
                    pass
                elif 'def convert_subset(set_name, config):' in line:
                    new_source.append("def convert_subset(set_name, output_dir, num_train, num_test, holes):\n")
                elif 'config.num_train' in line:
                    new_source.append(line.replace('config.num_train', 'num_train').replace('config.num_test', 'num_test'))
                elif 'config.holes' in line:
                    new_source.append(line.replace('config.holes', 'holes'))
                elif 'config.output_dir' in line:
                    new_source.append(line.replace('config.output_dir', 'output_dir'))
                elif '@cli.command(singleton=True)' in line:
                    pass
                elif 'def preprocess_data(config: DataProcessConfig):' in line:
                    pass
                elif 'convert_subset("train", config)' in line:
                    new_source.append("    convert_subset(\"train\", args.output_dir, args.num_train, args.num_test, args.holes)\n")
                elif 'convert_subset("test", config)' in line:
                    new_source.append("    convert_subset(\"test\", args.output_dir, args.num_train, args.num_test, args.holes)\n")
                elif 'cli()' in line:
                    new_source.append("    parser = argparse.ArgumentParser()\n")
                    new_source.append("    parser.add_argument(\"--output-dir\", type=str, default=\"data/sudoku-6x6\")\n")
                    new_source.append("    parser.add_argument(\"--num-train\", type=int, default=1000)\n")
                    new_source.append("    parser.add_argument(\"--num-test\", type=int, default=100)\n")
                    new_source.append("    parser.add_argument(\"--holes\", type=int, default=20)\n")
                    new_source.append("    args = parser.parse_args()\n")
                    new_source.append("    convert_subset(\"train\", args.output_dir, args.num_train, args.num_test, args.holes)\n")
                    new_source.append("    convert_subset(\"test\", args.output_dir, args.num_train, args.num_test, args.holes)\n")
                else:
                    new_source.append(line)
            cell['source'] = new_source

    with open('kaggle_finetune_6x6.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

def fix_9x9():
    with open('kaggle_finetune_9x9.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            new_source = []
            for line in cell['source']:
                if '!pip install -r requirements.txt' in line:
                    new_source.append("!pip install setuptools wheel\n")
                    new_source.append("!pip install einops tqdm pydantic wandb omegaconf hydra-core huggingface_hub peft argdantic coolname --quiet\n")
                else:
                    new_source.append(line)
            cell['source'] = new_source

    with open('kaggle_finetune_9x9.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

fix_6x6()
fix_9x9()
