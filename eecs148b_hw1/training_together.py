import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from eecs148b_hw1.transformer_lm import TransformerLM
from eecs148b_hw1.cross_entropy import cross_entropy_loss
from eecs148b_hw1.data_loading import get_batch
from eecs148b_hw1.experiment_log import ExperimentLogger



def parse_args():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--train-data", type=str, required=True, )
    parser.add_argument("--valid-data", type=str, required=True)
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints" )

    # Model hyperparameters
    parser.add_argument("--vocab-size",type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=2048)

    # Optimization hyperparameters
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=16000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-iters", type=int, default=50)

    # Runtime
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-ln", action="store_true")
    parser.add_argument("--no-pe", action="store_true")
    parser.add_argument("--overfit_debug", action="store_true")

    return parser.parse_args()



def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def lr_schedule(step: int, base_lr: float, warmup_steps: int):
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    return base_lr

     
def load_memmap_dataset(path: str):
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D token array in {path}, got shape {arr.shape}")
    return arr


def compute_loss(model, x, y):
    """
    x: (B, T)
    y: (B, T)
    logits: (B, T, vocab_size)
    """
    logits = model(x)
    vocab_size = logits.shape[-1]

    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = y.reshape(-1)

    loss = cross_entropy_loss(logits_flat, targets_flat)
    return loss



@torch.no_grad()
def evaluate(model, data, batch_size, context_length, device, eval_iters):
    model.eval()

    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(
            x=data,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )
        loss = compute_loss(model, x, y)
        losses.append(loss.item())

    mean_loss = sum(losses) / len(losses)
    perplexity = math.exp(mean_loss)
    return mean_loss, perplexity



def save_checkpoint(
    checkpoint_path: str,
    model,
    optimizer,
    step: int,
    train_loss: float,
    valid_loss: float,
    args,
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "train_loss": train_loss,
        "valid_loss": valid_loss,
        "args": vars(args),
    }
    torch.save(checkpoint, checkpoint_path)
    
    
def train(args, model, train_data, optimizer, valid_data, checkpoint_dir, logger):
    
    print(f"Device: {args.device}")
    
    best_valid_loss = float("inf")
    t0 = time.time()
    tokens_per_step = args.batch_size * args.context_length
    
    
    for step in range(1, args.max_steps + 1):
        model.train()
        
        lr = lr_schedule(step, args.learning_rate, args.warmup_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # get the data
        x, y = get_batch(
            x=train_data,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=args.device,
            )
        
        optimizer.zero_grad(set_to_none=True)

        loss = compute_loss(model, x, y)
        loss.backward()
        
        if args.grad_clip is not None and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
        optimizer.step()

        if step % args.eval_interval == 0 or step == 1:
            train_loss, train_ppl = evaluate(
                model=model,
                data=train_data,
                batch_size=args.batch_size,
                context_length=args.context_length,
                device=args.device,
                eval_iters=args.eval_iters,
            )

            valid_loss, valid_ppl = evaluate(
                model=model,
                data=valid_data,
                batch_size=args.batch_size,
                context_length=args.context_length,
                device=args.device,
                eval_iters=args.eval_iters,
            )
            
            elapsed = time.time() - t0
            tokens_processed = step * tokens_per_step
            
            print(
                f"step {step:6d} | "
                f"tokens={tokens_processed} | "
                f"train loss {train_loss:.4f} | train ppl {train_ppl:.4f} | "
                f"valid loss {valid_loss:.4f} | valid ppl {valid_ppl:.4f} | "
                f"time {elapsed:.1f}s | "
                f"lr={lr:.6f}"
            )
            
            logger.log(step, tokens_processed, "train", train_loss, train_ppl)
            logger.log(step, tokens_processed, "valid", valid_loss, valid_ppl)
            logger.save_curves()
            logger.write_summary()

            latest_ckpt = checkpoint_dir / "latest.pt"
            save_checkpoint(
                checkpoint_path=str(latest_ckpt),
                model=model,
                optimizer=optimizer,
                step=step,
                train_loss=train_loss,
                valid_loss=valid_loss,
                args=args,
            )

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_ckpt = checkpoint_dir / "best.pt"
                save_checkpoint(
                    checkpoint_path=str(best_ckpt),
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    train_loss=train_loss,
                    valid_loss=valid_loss,
                    args=args,
                )
        
        if step % args.save_interval == 0:
            ckpt_path = checkpoint_dir / f"step_{step}.pt"
            save_checkpoint(
                checkpoint_path=str(ckpt_path),
                model=model,
                optimizer=optimizer,
                step=step,
                train_loss=loss.item(),
                valid_loss=float("nan"),
                args=args,
            )
        
    final_ckpt = checkpoint_dir / "final.pt"
    save_checkpoint(
        checkpoint_path=str(final_ckpt),
        model=model,
        optimizer=optimizer,
        step=args.max_steps,
        train_loss=loss.item(),
        valid_loss=best_valid_loss,
        args=args,
    )   
    logger.save_curves()
    logger.write_summary()
    print("Training done!!.")
        
    
def train_overfit(args, model, train_data, optimizer, valid_data, checkpoint_dir, logger):   
    print(f"Device: {args.device}")
    
    print(f" train_data.shape {train_data.shape}, train_data.dtype{train_data.dtype}") #(541229348,), dtype: uint16
    print(f" train_data.min {int(train_data.min())}, train_data.max {int(train_data.max())}") # train_data.min 9, train_data.max 9999
    best_valid_loss = float("inf")
    t0 = time.time()
    tokens_per_step = args.batch_size * args.context_length
    
    # get the data
    # fixed tiny batch
    x_fixed, y_fixed = get_batch(
        train_data,
        batch_size=args.batch_size,
            context_length=args.context_length,
            device=args.device,
            )
    
    print("Fixed batch shape:", x_fixed.shape) #torch.Size([2, 16])
    print(f"x_fixed[0] {x_fixed[0]}") 
    print(f"y_fixed[0] {y_fixed[0]}")
    
    
    for step in range(1, args.max_steps + 1):
        model.train()
        
        lr = lr_schedule(step, args.learning_rate, args.warmup_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        logits = model(x_fixed)
        # print(f"logits.shape {logits.shape}") #torch.Size([2, 16, 10000])
        
        vocab_size = logits.shape[-1]

        loss = cross_entropy_loss(
            logits.reshape(-1, vocab_size),
            y_fixed.reshape(-1),
        )
        
        loss.backward()
        if step % 50 == 0:
            # ---- gradient norm check goes HERE ----
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.norm().item() ** 2

            total_norm = total_norm ** 0.5
            print("grad norm:", total_norm)

        optimizer.step()

        if step % 50 == 0:
            print(f"step {step}, loss {loss.item():.6f}")
            pred = torch.argmax(logits, dim=-1)

            print("input")
            print(x_fixed[0])

            print("target")
            print(y_fixed[0])

            print("pred")
            print(pred[0])
        


def main():
    args = parse_args()
    set_seed(args.seed)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    logger = ExperimentLogger(run_dir=str(run_dir), config=vars(args))
    
    print("Loading datasets with memory mapping...")
    train_data = load_memmap_dataset(args.train_data)
    valid_data = load_memmap_dataset(args.valid_data)
    
    print(f"Train shape: {train_data.shape}, dtype: {train_data.dtype}")
    print(f"Valid shape: {valid_data.shape}, dtype: {valid_data.dtype}")
    
    print("Checking token ranges...")
    train_min, train_max = int(train_data.min()), int(train_data.max())
    valid_min, valid_max = int(valid_data.min()), int(valid_data.max())
    print(f"Train token range: [{train_min}, {train_max}]")
    print(f"Valid token range: [{valid_min}, {valid_max}]")
    
    if train_min < 0 or valid_min < 0:
        raise ValueError("Found negative token IDs in dataset.")
    if train_max >= args.vocab_size or valid_max >= args.vocab_size:
        raise ValueError(
            f"Found token ID >= vocab_size ({args.vocab_size}). "
            f"Train max: {train_max}, Valid max: {valid_max}"
        )
        
        
    print("Building model...")
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        use_layernorm = not args.no_ln,
        use_position_embeddings = not args.no_pe,
        device=args.device,
        dtype=torch.float32,
    ).to(args.device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )
    
    
    # config_path = checkpoint_dir / "config.json"
    # with open(config_path, "w", encoding="utf-8") as f:
    #     json.dump(vars(args), f, indent=2)
        
    print("Starting training...")
    if args.overfit_debug:
        train_overfit(args, model, train_data, optimizer, valid_data, checkpoint_dir,logger )
    else:        
        train(args, model, train_data, optimizer, valid_data, checkpoint_dir, logger)
    
    
        

    
if __name__ == "__main__":
    main()