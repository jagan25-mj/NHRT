import os
import json
import random
import numpy as np
from tqdm import tqdm
from argdantic import ArgParser
from pydantic import BaseModel

try:
    from dataset.common import PuzzleDatasetMetadata
except ImportError:
    from common import PuzzleDatasetMetadata

cli = ArgParser()

class DataProcessConfig(BaseModel):
    output_dir: str = "data/sudoku-6x6"
    num_train: int = 1000
    num_test: int = 100
    holes: int = 20  # Number of blanks

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
                        if solve(board):
                            return True
                        board[r, c] = 0
                return False
    return True

def generate_6x6_board():
    board = np.zeros((6, 6), dtype=int)
    solve(board)
    return board

def generate_puzzles(num_puzzles, num_holes):
    inputs = []
    labels = []
    for _ in tqdm(range(num_puzzles), desc="Generating 6x6 puzzles"):
        solution = generate_6x6_board()
        puzzle = solution.copy()
        
        # Poke holes
        cells = [(r, c) for r in range(6) for c in range(6)]
        random.shuffle(cells)
        for r, c in cells[:num_holes]:
            puzzle[r, c] = 0
            
        inputs.append(puzzle)
        labels.append(solution)
    return inputs, labels

def convert_subset(set_name: str, config: DataProcessConfig):
    num_puzzles = config.num_train if set_name == "train" else config.num_test
    
    inputs, labels = generate_puzzles(num_puzzles, config.holes)
    
    results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
    
    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)
    
    for i, (inp, out) in enumerate(zip(inputs, labels)):
        # 0 becomes 1 (blank), 1-6 becomes 2-7
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
        seq_len=36,
        vocab_size=8,  # PAD + "0" + "1"..."6"
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=1,
        num_puzzle_identifiers=1,
        total_groups=len(results["group_indices"]) - 1,
        mean_puzzle_examples=1,
        sets=["all"]
    )

    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)
        
    for k, v in results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)

    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)

@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    convert_subset("train", config)
    convert_subset("test", config)

if __name__ == "__main__":
    cli()
