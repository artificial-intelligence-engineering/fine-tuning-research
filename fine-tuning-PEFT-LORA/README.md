# Parameter-Efficient Fine-Tuning with LoRA (PEFT-LoRA)

## Overview

Parameter-Efficient Fine-Tuning (PEFT) adapts a pre-trained LLM by updating only a tiny fraction of trainable parameters while keeping the base model weights frozen. **Low-Rank Adaptation (LoRA)** achieves this by decomposing the weight update $\Delta W$ of a frozen linear layer $W$ into two small trainable low-rank matrices $A$ and $B$ of rank $r$:

$$W' = W + \frac{\alpha}{r} AB$$

Because $A$ and $B$ can be merged back into $W$ after training, LoRA introduces **zero additional inference latency** while drastically reducing memory and optimizer state requirements.

## Reference Implementation

In *Build a Large Language Model (From Scratch)* by Sebastian Raschka (**Appendix E: Parameter-efficient fine-tuning with LoRA**, pp. 322–336, and **Exercise 7.4**), LoRA is built from scratch in PyTorch and applied to both tasks from Chapters 6 and 7:

- **Classification task (Appendix E):** `gpt2-small` (124M) on **SMS Spam Collection**.
- **Instruction task (Exercise 7.4):** `gpt2-medium` (355M) on `instruction-data.json`.

## Step-by-Step Workflow

1. **`LoRALayer(in_dim, out_dim, rank, alpha)`:**
   - Initializes matrix `A` (`in_dim × rank`) with `kaiming_uniform_` and matrix `B` (`rank × out_dim`) with **zeros** (`torch.zeros`), computing `alpha * (x @ A @ B)`.
   - Zero-initializing `B` ensures $AB = 0$ at step 0 so the pre-trained model's initial predictions remain unchanged.
2. **`LinearWithLoRA(linear, rank, alpha)`:**
   - Wraps an existing `nn.Linear` layer and returns `self.linear(x) + self.lora(x)`.
3. **`replace_linear_with_lora(model, rank, alpha)`:**
   - Recursively traverses `model.named_children()` to replace all `nn.Linear` layers (multi-head attention, feed-forward blocks, and output head) with `LinearWithLoRA`.
4. **Freezing & Training:**
   - Freezes all original parameters (`124,441,346 → 0` trainable parameters), then injects LoRA layers (`rank=16`, `alpha=16`), leaving only **2,666,528** trainable parameters (a **~50×** reduction).

## Key Practical Takeaways

- **High accuracy with minimal parameters:** On SMS spam classification (5 epochs, `AdamW`, `lr=5e-5`), LoRA reached **100.00% train / 96.64% validation / 98.00% test accuracy** while updating only 2.7M parameters.
- **Training speed vs. model scale:** On the small 124M model, LoRA took slightly longer (`12.10 min`) due to extra forward-pass operations. However, on `gpt2-medium` (355M) in Exercise 7.4 (Nvidia L4 GPU), LoRA was **~28% faster** (`1.30 min` vs. `1.80 min`) while matching the ~50 evaluation score.
- **Memory extensions:** In production workflows with larger models, **QLoRA** combines LoRA with 4-bit NormalFloat (`NF4`) quantization, double quantization, and paged optimizers to fit 65B-scale models on a single 48 GB GPU.
