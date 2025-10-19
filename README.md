# MiniFin-QA LM

MiniFin-QA LM is a compact finance-domain language model trained from scratch on earnings-call transcripts. The repository contains the full reproducible pipeline: corpus assembly, tokenizer training, model definition, pretraining script, evaluation artifacts, and sample generations.

---

## Highlights
- **Domain data pipeline** – aggregates raw transcripts from `dataset_finance/`, performs dialogue-preserving cleanup, filters noisy samples, and creates a reproducible 90/10 split (`data/train.txt`, `data/val.txt`).
- **Custom tokenizer** – trains an 8–12k byte-level BPE vocabulary with finance-friendly casing, saved under `tokenizer/`.
- **Mini GPT-style decoder** – 4 layers, 256 hidden size, 4 heads, 1,024 FFN width (≈5.5M params) implemented in pure PyTorch (`model.py`).
- **Training loop** – cosine LR with warmup, validation perplexity tracking, loss plots, and checkpointing; automatically prefers Apple MPS when available (`train.py`).
- **Artifacts** – `loss_plot.png`, `artifacts/model_state.pt`, `artifacts/training_history.json`, plus optional `samples.txt` generations for reporting.

---

## Repository Layout
```
project_root/
├─ dataset_finance/            # provided transcripts (unchanged)
├─ data/
│  ├─ train.txt                # generated cleaned corpus
│  └─ val.txt
├─ tokenizer/                  # byte-level BPE artifacts
│  ├─ tokenizer.json
│  ├─ vocab.json
│  ├─ merges.txt
│  └─ tokenizer_config.json
├─ artifacts/                  # training outputs (config, history, checkpoint)
├─ model.py                    # Mini GPT decoder definition
├─ train.py                    # training & generation CLI
├─ build_from_kaggle_calls.py  # corpus builder
├─ train_tokenizer.py          # tokenizer trainer
├─ loss_plot.png               # train/val loss curves
├─ samples.txt                 # generated samples (optional)
└─ requirements.txt
```

---

## Quickstart

1. **Install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Assemble the corpus**
   ```bash
   python build_from_kaggle_calls.py \
       --input_dir dataset_finance \
       --outdir data \
       --target_mb 8
   ```

3. **Train the tokenizer**
   ```bash
   python train_tokenizer.py \
       --files data/train.txt \
       --output_dir tokenizer \
       --vocab_size 9000
   ```

4. **Pretrain the model**
   ```bash
   python train.py \
       --data_dir data \
       --tokenizer_dir tokenizer \
       --output_dir artifacts \
       --batch_size 8 \
       --epochs 4 \
       --save_checkpoint
   ```

5. **Generate finance-toned samples**
   ```bash
   python train.py \
       --generate \
       --checkpoint_path artifacts/model_state.pt \
       --tokenizer_dir tokenizer \
       --prompt "Operator: Good day, and welcome to the earnings call." \
       --max_new_tokens 120 \
       --temperature 0.9 \
       --top_p 0.9 \
       --samples_file samples.txt
   ```

> **Note:** `train.py` automatically selects Apple MPS when present; otherwise it gracefully falls back to CPU.

---

## Training Results

| Epoch | Train Loss | Val Loss | Val Perplexity |
|-------|------------|----------|----------------|
| 1     | 5.33       | 4.40     | 81.52          |
| 2     | 4.20       | 4.04     | 57.03          |
| 3     | 3.95       | 3.90     | 49.23          |
| 4     | 3.84       | 3.87     | 47.73          |

- **Model size:** 5,524,992 parameters (`model.py` at default config).
- **Corpus size:** ~8.0 MB cleaned text (132 transcripts used).
- **Sequence length:** 256 tokens, batch size 8 on CPU.
- **Optimizer:** AdamW (`lr=3e-4`, `betas=(0.9, 0.95)`, weight decay `0.1`).
- **Scheduler:** cosine decay with 3% warmup.
- **Gradient clip:** 1.0.

Loss curves are available in `loss_plot.png`, generated automatically after training.

---

## Sample Generation

```
Prompt: Operator: Good day, and welcome to the earnings call.
Sample: Operator: Good day, and welcome to the earnings call. A D draw Ahmad, and have time, on-looking statements are based on a reminder, and uncertainties that are more leadership and uncertainties.

statements are important things that will be no duty to the companies' most recent SEC filings with any of these companies' most recent SEC filings.

may make sure that the forward-looking statements regarding a
contemplated in the forward-looking statements will be realized.
```

Additional samples can be appended to `samples.txt` for qualitative analysis and inclusion in the final report.

---

## Configuration Reference (`train.py`)
- `--layers`, `--d_model`, `--n_heads`, `--d_ff`, `--dropout`, `--seq_len` control the model architecture.
- `--epochs`, `--batch_size`, `--lr`, `--warmup_ratio`, `--grad_clip` configure training dynamics.
- `--mixed_precision` (default `none`) is available for CUDA runs; macOS MPS currently uses full precision.
- `--save_checkpoint` writes `artifacts/model_state.pt`; corresponding metadata stored in `artifacts/model_config.json` and `artifacts/training_history.json`.
- `--loss_plot` path (default `loss_plot.png`) controls where the curve visualization is saved.
- `--generate` mode enables controlled sampling via `--temperature`, `--top_k`, `--top_p`, and `--samples_file`.

---

## Reproducibility Notes
- Random seed defaults to **42** and is applied to Python `random`, NumPy, and PyTorch.
- The corpus builder shuffles documents with the same seed before enforcing the size cap, ensuring deterministic `train.txt` and `val.txt`.
- `requirements.txt` pins major dependencies (PyTorch, tokenizers, NumPy, Matplotlib) compatible with Apple silicon wheels.
- Matplotlib runs headlessly via the Agg backend, and cache directories are redirected to `.cache/` for sandbox compatibility.

---

## Deliverables Checklist
- [x] `build_from_kaggle_calls.py`
- [x] `train_tokenizer.py`
- [x] `model.py`
- [x] `train.py`
- [x] `tokenizer/` artifacts
- [x] `data/train.txt`, `data/val.txt`
- [x] `loss_plot.png`
- [x] `artifacts/` (config, history, checkpoint)
- [x] `samples.txt` (example generations)
- [ ] `report.pdf` (1-page summary – to be added)
- [ ] `Rollno_Name_Assignment1.zip` packaging (to be generated after final report)

---

## License

This project builds on the publicly available earnings-call transcripts dataset (see `dataset_finance/LICENSE`). All newly created code in this repository is released under the MIT License unless otherwise specified.
# generative-ai-finance-assignment-1
