#!/usr/bin/env python3
"""
Training script for the MiniFin-QA language model.

Supports causal language modeling (GPT-style) with command-line configuration,
validation perplexity tracking, loss plotting, and optional text generation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

CACHE_DIR = Path.cwd() / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

MPL_CACHE = Path.cwd() / ".matplotlib_cache"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer

from model import MiniGPTConfig, MiniGPTLM


class BlockDataset(Dataset):
    """Slices a long token sequence into contiguous non-overlapping blocks."""

    def __init__(self, token_ids: Iterable[int], block_size: int):
        token_list = list(token_ids)
        if len(token_list) <= block_size:
            raise ValueError(
                f"Token sequence of length {len(token_list)} is too short for block size {block_size}."
            )
        self.tokens = torch.tensor(token_list, dtype=torch.long)
        self.block_size = block_size
        self.length = (len(self.tokens) - 1) // block_size

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.block_size
        end = start + self.block_size
        x = self.tokens[start:end]
        y = self.tokens[start + 1 : end + 1]
        return x, y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or generate with the MiniFin-QA language model.")
    parser.add_argument("--data_dir", type=Path, default=Path("data"), help="Directory with train.txt and val.txt.")
    parser.add_argument("--tokenizer_dir", type=Path, default=Path("tokenizer"), help="Directory with tokenizer.json.")
    parser.add_argument("--output_dir", type=Path, default=Path("artifacts"), help="Directory to store training outputs.")
    parser.add_argument("--arch", type=str, default="clm", choices=["clm"], help="Architecture to train (clm only).")
    parser.add_argument("--layers", type=int, default=4, help="Number of transformer blocks.")
    parser.add_argument("--d_model", type=int, default=256, help="Model hidden size.")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--d_ff", type=int, default=1024, help="Feed-forward hidden dimension.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability.")
    parser.add_argument("--seq_len", type=int, default=256, help="Context window length.")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="AdamW weight decay.")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup steps as fraction of total steps.")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--log_interval", type=int, default=50, help="Steps between logging.")
    parser.add_argument("--mixed_precision", type=str, default="none", choices=["none", "fp16", "bf16"], help="Mixed precision mode (CUDA only).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--save_checkpoint", action="store_true", help="Save model weights after training.")
    parser.add_argument("--checkpoint_path", type=Path, default=None, help="Path to model checkpoint for generation or resuming.")
    parser.add_argument("--generate", action="store_true", help="Run generation mode instead of training.")
    parser.add_argument("--prompt", type=str, default="", help="Prompt text for generation mode.")
    parser.add_argument("--max_new_tokens", type=int, default=120, help="Maximum tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature for generation.")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling cutoff.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling threshold.")
    parser.add_argument("--samples_file", type=Path, default=Path("samples.txt"), help="File to append generated samples.")
    parser.add_argument("--loss_plot", type=Path, default=Path("loss_plot.png"), help="Path to save loss curve plot.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> Tuple[torch.device, str]:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        device_type = "mps"
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        device_type = "cuda"
    else:
        device = torch.device("cpu")
        device_type = "cpu"
    return device, device_type


def load_tokenizer(tokenizer_dir: Path) -> Tokenizer:
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Could not find tokenizer.json in {tokenizer_dir}.")
    return Tokenizer.from_file(str(tokenizer_path))


def encode_file(path: Path, tokenizer: Tokenizer) -> List[int]:
    text = path.read_text(encoding="utf-8")
    encoding = tokenizer.encode(text)
    return encoding.ids


def build_dataloaders(
    train_tokens: List[int],
    val_tokens: List[int],
    block_size: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = BlockDataset(train_tokens, block_size)
    val_dataset = BlockDataset(val_tokens, block_size)

    def collate(batch):
        inputs, targets = zip(*batch)
        return torch.stack(inputs), torch.stack(targets)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=collate,
        num_workers=num_workers,
    )
    return train_loader, val_loader


def get_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_loop(
    model: MiniGPTLM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    device_type: str,
    epochs: int,
    grad_clip: float,
    log_interval: int,
    mixed_precision: str,
    loss_plot: Path,
) -> Dict[str, List[float]]:
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_ppl": []}
    total_steps = len(train_loader) * epochs
    scaler = torch.cuda.amp.GradScaler(enabled=(mixed_precision == "fp16" and device_type == "cuda"))

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for step, (inputs, targets) in enumerate(train_loader, start=1):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)

            if mixed_precision != "none" and device_type == "cuda":
                autocast_dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
                with torch.cuda.amp.autocast(dtype=autocast_dtype):
                    _, loss = model(inputs, targets)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                _, loss = model(inputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            scheduler.step()
            running_loss += loss.item()

            global_step = (epoch - 1) * len(train_loader) + step
            if step % log_interval == 0 or step == 1 or step == len(train_loader):
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"Epoch {epoch}/{epochs} | Step {step}/{len(train_loader)} | "
                    f"Loss {loss.item():.4f} | LR {current_lr:.6f} | Global Step {global_step}/{total_steps}"
                )

        avg_train_loss = running_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, device, device_type, mixed_precision)
        val_ppl = math.exp(min(val_loss, 20))  # guard against overflow

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_ppl"].append(val_ppl)

        print(
            f"[Epoch {epoch}] Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Perplexity: {val_ppl:.2f}"
        )

    plot_losses(history["train_loss"], history["val_loss"], loss_plot)
    return history


@torch.no_grad()
def evaluate(
    model: MiniGPTLM,
    val_loader: DataLoader,
    device: torch.device,
    device_type: str,
    mixed_precision: str,
) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0

    use_autocast = device_type == "cuda" and mixed_precision != "none"

    for inputs, targets in val_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        if use_autocast:
            autocast_dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
            with torch.cuda.amp.autocast(dtype=autocast_dtype):
                _, loss = model(inputs, targets)
        else:
            _, loss = model(inputs, targets)
        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(1, total_batches)


def plot_losses(train_losses: List[float], val_losses: List[float], path: Path) -> None:
    plt.figure(figsize=(6, 4))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved loss plot to {path}")


def save_artifacts(
    output_dir: Path,
    config: MiniGPTConfig,
    model: MiniGPTLM,
    history: Dict[str, List[float]],
    save_checkpoint: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "model_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    if save_checkpoint:
        checkpoint_path = output_dir / "model_state.pt"
        torch.save({"model_state": model.state_dict()}, checkpoint_path)
        print(f"Saved model checkpoint to {checkpoint_path}")


def load_model_from_checkpoint(
    checkpoint_path: Path,
    tokenizer_dir: Path,
    device: torch.device,
) -> Tuple[MiniGPTLM, Tokenizer]:
    config_path = checkpoint_path.parent / "model_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing model_config.json next to {checkpoint_path}.")
    config_dict = json.loads(config_path.read_text(encoding="utf-8"))
    config = MiniGPTConfig(**config_dict)
    tokenizer = load_tokenizer(tokenizer_dir)

    model = MiniGPTLM(config)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state["model_state"])
    model.to(device)
    model.eval()
    return model, tokenizer


def run_generation(args: argparse.Namespace) -> None:
    if not args.checkpoint_path:
        raise SystemExit("--checkpoint_path is required for generation mode.")

    device, _ = select_device()
    model, tokenizer = load_model_from_checkpoint(args.checkpoint_path, args.tokenizer_dir, device)
    if not args.prompt:
        raise SystemExit("Please provide a --prompt for generation.")

    encoding = tokenizer.encode(args.prompt)
    input_ids = torch.tensor([encoding.ids], dtype=torch.long, device=device)
    generated_ids = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    output_text = tokenizer.decode(generated_ids[0].tolist())
    args.samples_file.parent.mkdir(parents=True, exist_ok=True)
    with args.samples_file.open("a", encoding="utf-8") as f:
        f.write(f"Prompt: {args.prompt}\n")
        f.write(f"Sample: {output_text}\n")
        f.write("=" * 80 + "\n")
    print(output_text)


def main() -> None:
    args = parse_args()

    if args.generate:
        run_generation(args)
        return

    set_seed(args.seed)
    device, device_type = select_device()
    print(f"Using device: {device} ({device_type})")

    if args.mixed_precision != "none" and device_type != "cuda":
        print("Mixed precision currently supported for CUDA devices only; falling back to full precision.")
        args.mixed_precision = "none"

    tokenizer = load_tokenizer(args.tokenizer_dir)
    train_path = args.data_dir / "train.txt"
    val_path = args.data_dir / "val.txt"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"Expected train.txt and val.txt in {args.data_dir}.")

    print("Encoding training corpus...")
    train_tokens = encode_file(train_path, tokenizer)
    print("Encoding validation corpus...")
    val_tokens = encode_file(val_path, tokenizer)

    train_loader, val_loader = build_dataloaders(
        train_tokens,
        val_tokens,
        args.seq_len,
        args.batch_size,
        args.num_workers,
    )

    config = MiniGPTConfig(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_seq_len=args.seq_len,
    )
    model = MiniGPTLM(config).to(device)

    param_count = model.count_parameters()
    print(f"Model parameter count: {param_count:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    total_steps = len(train_loader) * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    scheduler = get_scheduler(optimizer, warmup_steps, total_steps)

    history = train_loop(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        device_type,
        args.epochs,
        args.grad_clip,
        args.log_interval,
        args.mixed_precision,
        args.loss_plot,
    )

    save_artifacts(args.output_dir, config, model, history, args.save_checkpoint)
    print("Training complete.")


if __name__ == "__main__":
    main()
