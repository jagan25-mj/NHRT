import typing
from typing import List, Dict, Any
import yaml
import os
import copy
import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
import torch.distributed as dist
import pydantic
from omegaconf import OmegaConf

from finetune_lora import PretrainConfig, init_train_state, create_dataloader
from models.losses import IGNORE_LABEL_ID

class EvalConfig(pydantic.BaseModel):
    checkpoint: str
    save_outputs: List[str] = ["inputs", "labels", "puzzle_identifiers", "logits", "q_halt_logits", "q_continue_logits"]
    max_steps: int = 32
    rollback_threshold: float = 1.0 # If max(Q) drops by 1.0, rollback
    noise_std: float = 0.05

def launch():
    eval_cfg = EvalConfig(**OmegaConf.to_container(OmegaConf.from_cli()))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    RANK = 0
    WORLD_SIZE = 1

    with open(os.path.join(os.path.dirname(eval_cfg.checkpoint), "all_config.yaml"), "r") as f:
        config = PretrainConfig(**yaml.safe_load(f))
        config.eval_save_outputs = eval_cfg.save_outputs
        config.checkpoint_path = "./eval_outputs"
        config.load_checkpoint = None
        config.resume_checkpoint = None
        config.halt_max_steps = eval_cfg.max_steps # override to allow more steps

    eval_loader, eval_metadata = create_dataloader(config, "test", test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)

    train_state = init_train_state(config, eval_metadata, world_size=WORLD_SIZE)
    
    try:
        train_state.model.load_state_dict(torch.load(eval_cfg.checkpoint, map_location=device, weights_only=True), assign=True)
    except:
        train_state.model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in torch.load(eval_cfg.checkpoint, map_location=device, weights_only=True).items()}, assign=True)
    
    print ("Starting Latent Backtracking Evaluation (LBSP)")
    train_state.model.eval()
    
    total_correct = 0
    total_cells = 0
    total_exact = 0
    total_puzzles = 0
    total_rollbacks = 0
    
    with torch.no_grad():
        for set_name, batch, global_batch_size in eval_loader:
            batch = {k: v.cuda().float() if v.is_floating_point() else v.cuda() for k, v in batch.items()}
            
            with torch.device("cuda"):
                carry = train_state.model.initial_carry(batch)
                
            history = []
            
            for step_i in range(eval_cfg.max_steps):
                saved_carry = {
                    "z_H": carry.inner_carry.z_H.clone(),
                    "z_L": carry.inner_carry.z_L.clone(),
                    "steps": carry.steps.clone(),
                    "halted": carry.halted.clone()
                }
                
                carry, loss, metrics, outputs, all_finish = train_state.model(carry=carry, batch=batch, return_keys=["logits", "q_halt_logits", "q_continue_logits"])
                
                q_halt = outputs.get("q_halt_logits", None)
                q_continue = outputs.get("q_continue_logits", None)
                
                rollback_triggered = False
                q_max = None
                if q_halt is not None and q_continue is not None:
                    q_max = torch.maximum(q_halt, q_continue)
                    
                    if len(history) > 0:
                        prev_q_max = history[-1]["q_max"]
                        delta_q = q_max - prev_q_max
                        rollback_mask = (delta_q < -eval_cfg.rollback_threshold) & (~carry.halted)
                        
                        if rollback_mask.any():
                            rollback_triggered = True
                            total_rollbacks += rollback_mask.sum().item()
                            
                            # Apply rollback where mask is true
                            carry.inner_carry.z_H[rollback_mask] = history[-1]["carry"]["z_H"][rollback_mask]
                            carry.inner_carry.z_L[rollback_mask] = history[-1]["carry"]["z_L"][rollback_mask]
                                
                            # Apply perturbation to explore new latent path
                            noise_H = torch.randn_like(carry.inner_carry.z_H[rollback_mask]) * eval_cfg.noise_std
                            noise_L = torch.randn_like(carry.inner_carry.z_L[rollback_mask]) * eval_cfg.noise_std
                            carry.inner_carry.z_H[rollback_mask] += noise_H
                            carry.inner_carry.z_L[rollback_mask] += noise_L
                                
                            carry.steps[rollback_mask] = history[-1]["carry"]["steps"][rollback_mask]
                            carry.halted[rollback_mask] = history[-1]["carry"]["halted"][rollback_mask]
                
                if not rollback_triggered:
                    history.append({"carry": saved_carry, "q_max": q_max})
                
                if all_finish:
                    break
                    
            labels = batch["labels"]
            logits = outputs.get("logits", None)
            
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
                            
    print("\n" + "=" * 60)
    print(f"6x6 SUDOKU LBSP EVALUATION RESULTS")
    print("=" * 60)
    print(f"Cell Accuracy:    {total_correct}/{total_cells} = {100*total_correct/max(total_cells,1):.2f}%")
    print(f"Puzzle Accuracy:  {total_exact}/{total_puzzles} = {100*total_exact/max(total_puzzles,1):.2f}%")
    print(f"Total Rollbacks Triggered: {total_rollbacks}")
    print("=" * 60)

if __name__ == "__main__":
    launch()
