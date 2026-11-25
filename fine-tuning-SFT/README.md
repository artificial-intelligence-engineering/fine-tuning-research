# Supervised Instruction Fine-Tuning (SFT)

## Overview

Supervised Fine-Tuning (SFT), also referred to as instruction fine-tuning, trains a pre-trained foundation model on `(instruction, response)` pairs so that it interprets prompts as directives to follow rather than merely continuing raw text. It is the standard post-training stage for building helpful assistant and chat models.

## Reference Implementation

In *Build a Large Language Model (From Scratch)* by Sebastian Raschka (**Chapter 7: Fine-tuning to follow instructions**, pp. 204–249), SFT is implemented on `GPTModel`:

- **Base Model:** `gpt2-medium` (355M parameters: `emb_dim=1024`, `n_layers=24`, `n_heads=16`, `context_length=1024`). Larger capacity is needed compared to `gpt2-small` (124M) for reliable instruction following.
- **Dataset:** `instruction-data.json` (**1,100** `instruction` / `input` / `output` entries), split into **935 train (85%) / 55 validation (5%) / 110 test (10%)**. Chapter 7 (Exercise 7.3) also covers scaling to the **Stanford Alpaca** dataset (52,002 entries).

## Step-by-Step Workflow

1. **Prompt Formatting (`format_input`):**
   - Formats examples using an **Alpaca-style** template (`### Instruction:`, optional `### Input:`, and `### Response:`).
   - Also demonstrates the shorter **Phi-3-style** template (`<|user|>`, `<|assistant|>`).
2. **Dynamic Batching & Loss Masking (`InstructionDataset` + `custom_collate_fn`):**
   - Applies dynamic padding per batch (up to an optional `allowed_max_length=1024`) instead of padding the entire dataset to the global maximum.
   - Builds next-token shifted targets, keeps the first `<|endoftext|>` (`50256`) token as the end-of-response signal, and replaces subsequent padding tokens with **`-100`** (PyTorch's `cross_entropy` `ignore_index`).
3. **Training:**
   - Reuses the autoregressive training loop with `AdamW` (`lr=5e-5`, `weight_decay=0.1`), `batch_size=8`, and `num_epochs=2`.
4. **Automated Evaluation (LLM-as-a-Judge):**
   - Generates responses on the test split and scores them automatically using **Llama 3 8B** served locally via **Ollama** (`ollama run llama3`), achieving an average score above **50/100**.

## Key Practical Takeaways

- **Prompt template efficiency:** Switching from the Alpaca template to the more compact Phi-3 template accelerated training by **~17%** while maintaining a similar evaluation score (~50).
- **Instruction masking:** Masking instruction tokens with `-100` in the target loss (Exercise 7.2) lowered the evaluation score by **~4 points** compared to computing loss over the full formatted sequence.
- **Memory scaling:** For larger datasets such as Stanford Alpaca on consumer GPUs, reduce `batch_size` (e.g., `4`, `2`, or `1`) and `allowed_max_length` (e.g., `512` or `256`).
