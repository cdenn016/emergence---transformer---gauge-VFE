"""
Train Standard Transformer Baseline
====================================

Fair comparison to gauge-theoretic transformer:
    - Same data (character-level WikiText-2)
    - Same parameter budget (~5,334 params)
    - Same training steps (20 for debug, 100 for standard)
    - Standard dot-product attention (NO gauge theory, NO KL divergence)

This answers the critical question:
    "Does SO(3) gauge structure help or hurt performance?"

Expected outcomes:
    1. Standard > Gauge: SO(3) is wrong inductive bias
    2. Standard = Gauge: SO(3) is neutral
    3. Standard < Gauge: SO(3) helps! (validates framework)

Author: Ablation study baseline
Date: November 2025
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

# Import standard transformer
from transformer.standard_transformer import StandardTransformerLM

# Import data loader (reuse from gauge model)
from transformer.data import create_char_dataloaders


def train_standard_baseline(
    config: dict,
    train_loader,
    val_loader,
    device: str = 'cpu',
    checkpoint_dir: str = 'checkpoints_publication/standard_baseline',
):
    """
    Train standard transformer baseline.

    Args:
        config: Model configuration
        train_loader: Training dataloader
        val_loader: Validation dataloader
        device: Device to train on
        checkpoint_dir: Where to save checkpoints
    """
    # Create checkpoint directory
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Create model
    print("\n" + "="*70)
    print("CREATING STANDARD TRANSFORMER")
    print("="*70)

    model = StandardTransformerLM(config).to(device)

    # Count parameters
    param_counts = model.count_parameters()
    print("\nParameter Breakdown:")
    for name, count in param_counts.items():
        print(f"  {name:20s}: {count:6d}")

    # Optimizer - use SAME learning rate as gauge model for fair comparison
    # Note: Standard transformers typically use lr=3e-4, but gauge model uses higher lr
    # We'll use standard lr first, then optionally try gauge model's higher lr
    lr = config.get('lr', 0.001)  # Standard transformer lr
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=config.get('weight_decay', 0.01),
        betas=(0.9, 0.999),
    )

    print(f"\nOptimizer: AdamW")
    print(f"  Learning rate: {lr}")
    print(f"  Weight decay:  {config.get('weight_decay', 0.01)}")

    # Training loop
    max_steps = config['max_steps']
    log_interval = config.get('log_interval', 5)
    eval_interval = config.get('eval_interval', 20)

    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    print(f"Max steps: {max_steps}")
    print(f"Device: {device}")

    model.train()
    step = 0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    start_time = time.time()

    pbar = tqdm(total=max_steps, desc="Training")

    train_iter = iter(train_loader)

    while step < max_steps:
        step += 1
        step_start = time.time()

        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # Unpack batch (dataloader returns tuple, not dict)
        input_ids, _ = batch
        input_ids = input_ids.to(device)
        labels = input_ids.clone()  # Next-token prediction uses same sequence

        # Forward pass
        optimizer.zero_grad()
        output = model(input_ids, labels=labels)
        loss = output['loss']

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('grad_clip', 1.0))
        optimizer.step()

        step_time = time.time() - step_start
        train_losses.append(loss.item())

        # Compute perplexity
        ppl = torch.exp(loss).item()

        # Update progress bar
        pbar.update(1)
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'ppl': f'{ppl:.1f}',
            'lr': f'{lr:.2e}',
            'time': f'{step_time:.2f}s'
        })

        # Logging
        if step % log_interval == 0 or step == max_steps:
            print(f"\nStep {step}/{max_steps} | Loss: {loss.item():.4f} | PPL: {ppl:.1f} | LR: {lr:.2e} | Time: {step_time:.2f}s")

        # Validation
        if step % eval_interval == 0 or step == max_steps:
            model.eval()
            val_loss = 0.0
            n_val_batches = min(100, len(val_loader))  # Limit validation batches

            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    if i >= n_val_batches:
                        break

                    # Unpack batch (dataloader returns tuple, not dict)
                    input_ids, _ = batch
                    input_ids = input_ids.to(device)
                    labels = input_ids.clone()

                    output = model(input_ids, labels=labels)
                    val_loss += output['loss'].item()

            val_loss /= n_val_batches
            val_ppl = np.exp(val_loss)
            val_losses.append(val_loss)

            print(f"  Validation @ step {step}:")
            print(f"    Loss: {val_loss:.4f}")
            print(f"    Perplexity: {val_ppl:.2f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = Path(checkpoint_dir) / 'best_model.pt'
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_ppl': val_ppl,
                    'config': config,
                }, checkpoint_path)
                print(f"\n  💾 Saving best model: {checkpoint_path}")

            model.train()

    pbar.close()

    # Training complete
    total_time = time.time() - start_time
    avg_step_time = total_time / max_steps

    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Average step time: {avg_step_time:.2f} seconds")
    print(f"Steps per second: {1/avg_step_time:.2f}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("="*70)

    # Compute improvement over random
    random_ppl = config['vocab_size']  # Uniform distribution
    final_ppl = np.exp(best_val_loss)
    improvement = random_ppl / final_ppl

    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE!")
    print("="*70)
    print(f"Final Loss: {best_val_loss:.4f}")
    print(f"Final Perplexity: {final_ppl:.2f}")
    print(f"\nImprovement over random: {improvement:.1f}x")

    # Save training log
    log_path = Path(checkpoint_dir) / 'training_log.json'
    log_data = {
        'config': config,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'best_val_ppl': final_ppl,
        'total_time_seconds': total_time,
        'improvement_over_random': improvement,
    }

    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)

    print(f"  💾 Saved training log: {log_path}")

    return model, best_val_loss, final_ppl


def main():
    parser = argparse.ArgumentParser(description='Train standard transformer baseline')
    parser.add_argument('--config', type=str, default='debug',
                        choices=['debug', 'debug_matched_lr', 'debug_moderate_lr', 'convergence_test', 'standard', 'extended'],
                        help='''Configuration options:
                        debug: 20 steps, lr=0.001 (UNFAIR - too low)
                        debug_matched_lr: 20 steps, lr=0.05 (FAIR - matches gauge)
                        debug_moderate_lr: 20 steps, lr=0.01 (moderate boost)
                        convergence_test: 1000 steps, lr=0.001 (long run)
                        standard: 100 steps
                        extended: 200 steps''')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides config default)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Device to train on')
    args = parser.parse_args()

    # Configuration
    configs = {
        'debug': {
            # FAIR COMPARISON: Match gauge model's learning rate
            'vocab_size': 256,
            'embed_dim': 11,
            'n_layers': 2,
            'n_heads': 1,
            'hidden_dim': 44,
            'max_seq_len': 32,
            'dropout': 0.1,
            'tie_embeddings': True,
            'batch_size': 16,
            'max_steps': 20,
            'lr': 0.05,  # MATCH gauge model!
            'weight_decay': 0.01,
            'grad_clip': 1.0,
            'log_interval': 5,
            'eval_interval': 20,
        },
        'debug_matched_lr': {
            # FAIR COMPARISON: Match gauge model's learning rate
            'vocab_size': 256,
            'embed_dim': 11,
            'n_layers': 2,
            'n_heads': 1,
            'hidden_dim': 44,
            'max_seq_len': 32,
            'dropout': 0.1,
            'tie_embeddings': True,
            'batch_size': 16,
            'max_steps': 20,
            'lr': 0.05,  # MATCH gauge model!
            'weight_decay': 0.01,
            'grad_clip': 1.0,
            'log_interval': 5,
            'eval_interval': 20,
        },
        'debug_moderate_lr': {
            # MODERATE: 10× higher than standard, still conservative
            'vocab_size': 256,
            'embed_dim': 11,
            'n_layers': 2,
            'n_heads': 1,
            'hidden_dim': 44,
            'max_seq_len': 32,
            'dropout': 0.1,
            'tie_embeddings': True,
            'batch_size': 16,
            'max_steps': 20,
            'lr': 0.01,  # Moderate boost
            'weight_decay': 0.01,
            'grad_clip': 1.0,
            'log_interval': 5,
            'eval_interval': 20,
        },
        'convergence_test': {
            # LONG RUN: Let standard model converge with low lr
            'vocab_size': 256,
            'embed_dim': 11,
            'n_layers': 2,
            'n_heads': 1,
            'hidden_dim': 44,
            'max_seq_len': 32,
            'dropout': 0.1,
            'tie_embeddings': True,
            'batch_size': 16,
            'max_steps': 1000,  # 50× more steps
            'lr': 0.001,
            'weight_decay': 0.01,
            'grad_clip': 1.0,
            'log_interval': 50,
            'eval_interval': 200,
        },
        'standard': {
            'vocab_size': 256,
            'embed_dim': 11,
            'n_layers': 2,
            'n_heads': 1,
            'hidden_dim': 44,
            'max_seq_len': 32,
            'dropout': 0.1,
            'tie_embeddings': True,
            'batch_size': 32,
            'max_steps': 100,
            'lr': 0.001,
            'weight_decay': 0.01,
            'grad_clip': 1.0,
            'log_interval': 10,
            'eval_interval': 50,
        },
        'extended': {
            'vocab_size': 256,
            'embed_dim': 11,
            'n_layers': 3,
            'n_heads': 1,
            'hidden_dim': 44,
            'max_seq_len': 32,
            'dropout': 0.1,
            'tie_embeddings': True,
            'batch_size': 32,
            'max_steps': 200,
            'lr': 0.001,
            'weight_decay': 0.01,
            'grad_clip': 1.0,
            'log_interval': 10,
            'eval_interval': 50,
        },
    }

    config = configs[args.config]

    # Override lr if specified
    if args.lr is not None:
        config['lr'] = args.lr

    print("="*70)
    print("STANDARD TRANSFORMER BASELINE EXPERIMENT")
    print("="*70)
    print(f"\nConfiguration: {args.config}")
    print(f"\nModel:")
    print(f"  vocab_size:   {config['vocab_size']}")
    print(f"  embed_dim:    {config['embed_dim']}")
    print(f"  n_layers:     {config['n_layers']}")
    print(f"  n_heads:      {config['n_heads']}")
    print(f"  hidden_dim:   {config['hidden_dim']}")
    print(f"  max_seq_len:  {config['max_seq_len']}")
    print(f"\nTraining:")
    print(f"  batch_size:   {config['batch_size']}")
    print(f"  max_steps:    {config['max_steps']}")
    print(f"  lr:           {config['lr']}")
    print(f"  weight_decay: {config['weight_decay']}")

    # Load data
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)

    train_loader, val_loader, actual_vocab_size = create_char_dataloaders(
        batch_size=config['batch_size'],
        max_seq_len=config['max_seq_len'],
        num_workers=0,
    )

    # Update vocab size based on actual data
    config['vocab_size'] = actual_vocab_size
    print(f"\nActual vocabulary size: {config['vocab_size']}")

    # Train
    model, best_val_loss, best_val_ppl = train_standard_baseline(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        checkpoint_dir=f'checkpoints_publication/standard_baseline_{args.config}',
    )

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE!")
    print("="*70)
    print(f"\nStandard Transformer Results:")
    print(f"  Validation Loss: {best_val_loss:.4f}")
    print(f"  Validation PPL:  {best_val_ppl:.2f}")
    print(f"\nCompare to Gauge Model:")
    print(f"  Gauge (learned):          PPL ≈ 23.51")
    print(f"  Gauge (variational_approx): PPL ≈ 23.18")
    print(f"  Gauge (variational_full):   PPL ≈ 23.11")
    print(f"  Standard (this run):        PPL = {best_val_ppl:.2f}")

    if best_val_ppl < 23.11:
        print(f"\n🔴 Standard transformer BETTER than gauge model!")
        print(f"   → SO(3) structure may be HURTING performance")
    elif best_val_ppl > 24.0:
        print(f"\n🟢 Gauge model BETTER than standard transformer!")
        print(f"   → SO(3) structure is HELPING performance")
    else:
        print(f"\n🟡 Performance is COMPARABLE")
        print(f"   → SO(3) structure is neutral (neither helps nor hurts)")

    print("="*70)


if __name__ == '__main__':
    main()