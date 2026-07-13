
import random
import pandas as pd

BASE = [
    [1,2,3,4,5,6],
    [4,5,6,1,2,3],
    [2,3,4,5,6,1],
    [5,6,1,2,3,4],
    [3,4,5,6,1,2],
    [6,1,2,3,4,5],
]

def permute(grid):
    nums=list(range(1,7))
    random.shuffle(nums)
    mp={i+1:nums[i] for i in range(6)}
    return [[mp[v] for v in row] for row in grid]

def make_puzzle(sol, clues=18):
    cells=[(r,c) for r in range(6) for c in range(6)]
    random.shuffle(cells)
    keep=set(cells[:clues])
    puz=[]
    for r in range(6):
        row=[]
        for c in range(6):
            row.append(sol[r][c] if (r,c) in keep else 0)
        puz.append(row)
    return puz

def encode(g):
    return "".join(str(x) for row in g for x in row)

def generate_dataset(n=10000, clues=18):
    rows=[]
    for i in range(n):
        sol=permute(BASE)
        puz=make_puzzle(sol, clues)
        rows.append({
            "id": i,
            "puzzle": encode(puz),
            "solution": encode(sol),
            "clues": clues
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df=generate_dataset(100000,18)
    df.to_csv("sudoku6x6_train.csv", index=False)
    print("saved", len(df), "samples")
