import json

def fix_6x6():
    with open('kaggle_finetune_6x6.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            
            # 1. Change git clone url
            if 'https://github.com/sapientinc/HRM.git' in source:
                source = source.replace('https://github.com/sapientinc/HRM.git', 'https://github.com/jagan25-mj/NHRT.git')
            
            # 2. Fix the dataset builder recursion bug
            if 'dataset/build_6x6_sudoku_dataset.py' in source:
                # We will entirely rewrite the source of this cell to be perfectly correct
                new_source = '''%%writefile dataset/build_6x6_sudoku_dataset.py
import os
import json
import random
import numpy as np
from tqdm import tqdm
import argparse

try:
    from dataset.common import PuzzleDatasetMetadata
except ImportError:
    from common import PuzzleDatasetMetadata

def is_valid(board, r, c, val):
    if val in board[r]: return False
    if val in board[:, c]: return False
    br, bc = 2 * (r // 2), 3 * (c // 3)
    if val in board[br:br+2, bc:bc+3]: return False
    return True

def solve(board):
    for r in range(6):
        for c in range(6):
            if board[r, c] == 0:
                nums = list(range(1, 7))
                random.shuffle(nums)
                for val in nums:
                    if is_valid(board, r, c, val):
                        board[r, c] = val
                        if solve(board): return True
                        board[r, c] = 0
                return False
    return True

def generate_6x6_board():
    board = np.zeros((6, 6), dtype=int)
    solve(board)
    return board

def generate_puzzles(num_puzzles, num_holes):
    inputs, labels = [], []
    for _ in tqdm(range(num_puzzles), desc="Generating 6x6 puzzles"):
        solution = generate_6x6_board()
        puzzle = solution.copy()
        cells = [(r, c) for r in range(6) for c in range(6)]
        random.shuffle(cells)
        for r, c in cells[:num_holes]:
            puzzle[r, c] = 0
        inputs.append(puzzle)
        labels.append(solution)
    return inputs, labels

def convert_subset(set_name, output_dir, num_train, num_test, holes):
    num_puzzles = num_train if set_name == "train" else num_test
    inputs, labels = generate_puzzles(num_puzzles, holes)
    results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)
    for i, (inp, out) in enumerate(zip(inputs, labels)):
        results["inputs"].append(inp.flatten() + 1)
        results["labels"].append(out.flatten() + 1)
        results["puzzle_indices"].append(i + 1)
        results["puzzle_identifiers"].append(0)
        results["group_indices"].append(i + 1)
    results = {
        "inputs": np.array(results["inputs"], dtype=np.int32),
        "labels": np.array(results["labels"], dtype=np.int32),
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }
    metadata = PuzzleDatasetMetadata(
        seq_len=36, vocab_size=8, pad_id=0, ignore_label_id=0,
        blank_identifier_id=1, num_puzzle_identifiers=1,
        total_groups=len(results["group_indices"]) - 1,
        mean_puzzle_examples=1, sets=["all"]
    )
    save_dir = os.path.join(output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)
    for k, v in results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)
    with open(os.path.join(output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="data/sudoku-6x6")
    parser.add_argument("--num-train", type=int, default=1000)
    parser.add_argument("--num-test", type=int, default=100)
    parser.add_argument("--holes", type=int, default=20)
    args = parser.parse_args()
    
    convert_subset("train", args.output_dir, args.num_train, args.num_test, args.holes)
    convert_subset("test", args.output_dir, args.num_train, args.num_test, args.holes)
'''
                source = new_source
                
            cell['source'] = [line + '\n' for line in source.split('\n')]
            # Fix double newlines at the end of lines
            cell['source'] = [line.replace('\n\n', '\n') for line in cell['source']]
            if cell['source'] and cell['source'][-1].endswith('\n'):
                cell['source'][-1] = cell['source'][-1][:-1]
                
    with open('kaggle_finetune_6x6.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

def fix_9x9():
    with open('kaggle_finetune_9x9.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'https://github.com/sapientinc/HRM.git' in source:
                source = source.replace('https://github.com/sapientinc/HRM.git', 'https://github.com/jagan25-mj/NHRT.git')
                
            cell['source'] = [line + '\n' for line in source.split('\n')]
            # Fix double newlines
            cell['source'] = [line.replace('\n\n', '\n') for line in cell['source']]
            if cell['source'] and cell['source'][-1].endswith('\n'):
                cell['source'][-1] = cell['source'][-1][:-1]

    with open('kaggle_finetune_9x9.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

fix_6x6()
fix_9x9()
