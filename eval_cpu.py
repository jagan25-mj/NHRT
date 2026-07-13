import os, glob, yaml
import torch
import numpy as np

os.environ["DISABLE_COMPILE"] = "1"

from pretrain import PretrainConfig, init_train_state, create_dataloader
from models.losses import IGNORE_LABEL_ID

# Find the latest checkpoint
latest_step = -1
best_path = None
for root, dirs, files in os.walk("checkpoints"):
    for f in files:
        if f.startswith("step_") and "all_preds" not in f:
            try:
                step = int(f.split("_")[1])
                if step > latest_step:
                    latest_step = step
                    best_path = os.path.join(root, f)
            except: pass

if not best_path:
    print("ERROR: No checkpoint found!")
    exit(1)

print(f"Checkpoint: {best_path} (step {latest_step})")

# Load config from checkpoint
with open(os.path.join(os.path.dirname(best_path), "all_config.yaml"), "r") as f:
    config = PretrainConfig(**yaml.safe_load(f))
config.eval_save_outputs = ["logits"]
config.checkpoint_path = os.path.dirname(best_path)

# Create dataloader
eval_loader, eval_metadata = create_dataloader(config, "test", test_set_mode=True, 
    epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=0, world_size=1)

# Create model on CPU
train_state = init_train_state(config, eval_metadata, world_size=1)

# Load checkpoint weights to CPU
try:
    train_state.model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=True), assign=True)
except:
    train_state.model.load_state_dict(
        {k.removeprefix("_orig_mod."): v for k, v in torch.load(best_path, map_location="cpu", weights_only=True).items()}, 
        assign=True)

train_state.model = train_state.model.float().cpu()
train_state.model.eval()

total_correct = 0
total_cells = 0
total_exact = 0
total_puzzles = 0
max_iters = 16

print("Running evaluation on CPU...")

with torch.inference_mode():
    for set_name, batch, global_batch_size in eval_loader:
        batch = {k: v.cpu().float() if v.is_floating_point() else v.cpu() for k, v in batch.items()}
        
        with torch.device("cpu"):
            carry = train_state.model.initial_carry(batch)
        
        for step_i in range(max_iters):
            try:
                carry, _, metrics, preds, all_finish = train_state.model(
                    carry=carry, batch=batch, return_keys=["logits"])
                if all_finish:
                    break
            except Exception as e:
                print(f"  Iteration {step_i} failed with error: {e}")
                import traceback
                traceback.print_exc()
                exit(1)
        
        labels = batch["labels"]
        logits = preds.get("logits", None)
        
        if logits is not None:
            predictions = torch.argmax(logits, dim=-1)
            mask = labels != IGNORE_LABEL_ID
            
            correct = (predictions == labels) & mask
            total_correct += correct.sum().item()
            total_cells += mask.sum().item()
            
            for i in range(labels.shape[0]):
                puzzle_mask = mask[i]
                if puzzle_mask.sum() > 0:
                    total_puzzles += 1
                    if correct[i].sum() == puzzle_mask.sum():
                        total_exact += 1
        
        break # Just do one batch for debugging

print("=" * 60)
print(f"RESULTS (step {latest_step})")
print("=" * 60)
print(f"Cell Accuracy:    {total_correct}/{total_cells} = {100*total_correct/max(total_cells,1):.2f}%")
print(f"Puzzle Accuracy:  {total_exact}/{total_puzzles} = {100*total_exact/max(total_puzzles,1):.2f}%")
