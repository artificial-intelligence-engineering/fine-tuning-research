# Classification Fine-Tuning

## Overview

Classification fine-tuning adapts a pre-trained Large Language Model (LLM) to assign discrete class labels to input sequences (e.g., spam detection, sentiment analysis, or topic categorization). Rather than generating free-form text across the full vocabulary, the model's language-modeling output head is replaced with a task-specific classification head.

This directory provides a self-contained, reproducible PyTorch implementation based on **Chapter 6 (*Fine-tuning for classification*)** of *Build a Large Language Model (From Scratch)* by Sebastian Raschka.

---

## Project Structure

```text
fine-tuning-classification/
├── .gitignore              # Ignores downloaded datasets, GPT-2 weights, and outputs
├── .python-version         # Pins Python 3.12 for uv reproducibility
├── pyproject.toml          # Project metadata and dependencies
├── uv.lock                 # Locked dependency versions for reproducible installs
├── README.md               # This guide
├── src/
│   ├── __init__.py
│   ├── dataset.py          # UCI SMS Spam download, balancing, splitting, and SpamDataset
│   ├── gpt_download.py     # OpenAI GPT-2 TensorFlow checkpoint downloader & parser
│   ├── gpt_model.py        # GPTModel architecture, weight loader, and head/layer freezing
│   └── train.py            # Training loop, evaluation, plotting, and CLI entrypoint
└── tests/
    └── test_classification.py  # Unit and smoke tests (offline, no downloads required)
```

---

## Step-by-Step Installation (Reproducible with `uv`)

### 1. Prerequisites
- **Git**
- **[uv](https://docs.astral.sh/uv/)** (`>= 0.5.0` recommended):
  ```bash
  # Linux / macOS
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

### 2. Clone the Repository and Enter the Directory
```bash
git clone git@github.com:artificial-intelligence-engineering/fine-tuning-research.git
cd fine-tuning-research/fine-tuning-classification
```

### 3. Install Python and Locked Dependencies
`uv` automatically reads `.python-version` (`3.12`) and `uv.lock`, creates `.venv/`, and installs exact package versions:
```bash
uv sync
```

---

## How to Test and Run

### 1. Run Unit & Smoke Tests (Offline)
Runs fast offline tests covering dataset balancing/splitting, `SpamDataset` padding/truncation, layer freezing, loss calculation, and inference:
```bash
uv run pytest -v
```

### 2. Run End-to-End Pipeline in Test Mode (CPU, Fast)
Downloads and prepares the **SMS Spam Collection** dataset, then runs 1 epoch on a tiny `GPTModel` on CPU to verify the entire pipeline without downloading GPT-2 weights:
```bash
uv run python -m src.train --test_mode --epochs 1 --no_plots
```

### 3. Run Full Classification Fine-Tuning (`gpt2-small` 124M)
Downloads the pre-trained OpenAI `gpt2-small (124M)` checkpoint into `gpt2/`, freezes base layers (training only `trf_blocks[-1]`, `final_norm`, and the `Linear(768, 2)` classification head), trains for 5 epochs with `AdamW` (`lr=5e-5`), saves `outputs/review_classifier.pth`, and generates `outputs/loss-plot.pdf` and `outputs/accuracy-plot.pdf`:
```bash
uv run python -m src.train
```

Optional CLI flags:
- `--epochs 5` — Number of training epochs
- `--batch_size 8` — DataLoader batch size
- `--lr 5e-5` — AdamW learning rate
- `--data_dir data` — Directory for downloaded/split CSV files
- `--models_dir gpt2` — Directory for GPT-2 checkpoint files
- `--output_dir outputs` — Directory for saved `.pth` weights and `.pdf` plots
- `--no_plots` — Skip generating PDF plots

---

## Reference Details & Key Takeaways

- **Base Model:** `gpt2-small` (124M parameters) initialized with pre-trained OpenAI GPT-2 weights.
- **Dataset:** **SMS Spam Collection** (`SMSSpamCollection.tsv`, UCI), class-balanced between `"ham"` (`0`) and `"spam"` (`1`), split into **70% train / 10% validation / 20% test**.
- **Last-Token Classification:** Because GPT uses causal attention masking, only the final token (`[:, -1, :]`) attends to every preceding token in the sequence. Cross-entropy loss and predictions (`softmax` + `argmax`) are computed exclusively on this position.
- **Padding length matters:** Padding all sequences to `max_length=1024` instead of the dataset's actual max length degraded test accuracy from **95.67% to 78.33%**.
- **Partial vs. full layer training:** Unfreezing all layers instead of only `trf_blocks[-1]`, `final_norm`, and `out_head` yielded only a ~1% gain (**95.67% → 96.67%**) at significantly higher compute cost.
- **Token position:** Classifying from the first token (`[:, 0, :]`) instead of the last token dropped test accuracy to **75.00%**, confirming the effect of causal attention.
