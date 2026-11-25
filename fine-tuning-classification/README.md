# Classification Fine-Tuning

## Overview

Classification fine-tuning adapts a pre-trained Large Language Model (LLM) to assign discrete class labels to input sequences (e.g., spam detection, sentiment analysis, or topic categorization). Rather than generating free-form text across the full vocabulary, the model's language-modeling output head is replaced with a task-specific classification head.

## Reference Implementation

In *Build a Large Language Model (From Scratch)* by Sebastian Raschka (**Chapter 6: Fine-tuning for classification**, pp. 169–203), this technique is demonstrated step by step on `GPTModel`:

- **Base Model:** `gpt2-small` (124M parameters) initialized with pre-trained OpenAI GPT-2 weights.
- **Dataset:** **SMS Spam Collection** (`SMSSpamCollection.tsv`, UCI), class-balanced between `"ham"` (`0`) and `"spam"` (`1`), split into **70% train / 10% validation / 20% test**.

## Step-by-Step Workflow

1. **Tokenization & Padding (`SpamDataset`):**
   - Tokenizes text using `tiktoken` (GPT-2 BPE).
   - Pads sequences with the `<|endoftext|>` token (`50256`) up to the longest message in the training split (120 tokens in the reference example), using a batch size of `8`.
2. **Architecture Modification & Partial Freezing:**
   - Replaces `out_head` (originally `768 -> 50,257` vocabulary logits) with a binary classification head `Linear(768, 2)`.
   - Freezes all base parameters (`requires_grad = False`) except the new `out_head`, the final transformer block (`trf_blocks[-1]`), and `final_norm`.
3. **Last-Token Classification:**
   - Because GPT uses causal attention masking, only the final token (`[:, -1, :]`) attends to every preceding token in the sequence.
   - Cross-entropy loss and predictions (`softmax` + `argmax`) are computed exclusively on this last-token representation.
4. **Training & Evaluation:**
   - Trained with `AdamW`, improving from near-random baseline accuracy (~45–48%) to **95.67% test accuracy**.

## Key Practical Takeaways

- **Padding length matters:** Padding all sequences to `max_length=1024` instead of the dataset's actual max length degraded test accuracy from **95.67% to 78.33%**.
- **Partial vs. full layer training:** Unfreezing all layers instead of only the last transformer block, `final_norm`, and `out_head` yielded only a ~1% gain (**95.67% → 96.67%**) at significantly higher compute cost.
- **Token position:** Classifying from the first token (`[:, 0, :]`) instead of the last token dropped test accuracy to **75.00%**, confirming the effect of causal attention.
