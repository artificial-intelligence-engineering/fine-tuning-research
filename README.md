# fine-tuning-research

Research on Fine-Tuning and Adaptation of Foundation LLMs.

## Fine-Tuning Techniques

This repository explores the core practical fine-tuning techniques for Large Language Models, grounded in *Build a Large Language Model (From Scratch)* by Sebastian Raschka and broader AI engineering practices:

- **[Classification Fine-Tuning](./fine-tuning-classification/README.md)** (`fine-tuning-classification/`): Adapts a pre-trained GPT model (`gpt2-small`, 124M) for discrete sequence classification by replacing the vocabulary output head with a task-specific linear head, freezing earlier layers, and computing loss on the final causal token.
- **[Supervised Instruction Fine-Tuning (SFT)](./fine-tuning-SFT/README.md)** (`fine-tuning-SFT/`): Teaches a pre-trained model (`gpt2-medium`, 355M) to follow natural-language instructions using Alpaca/Phi-3 prompt templates, dynamic batch padding, `-100` target masking, and automated LLM-as-a-judge evaluation.
- **[PEFT with LoRA](./fine-tuning-PEFT-LORA/README.md)** (`fine-tuning-PEFT-LORA/`): Implements Low-Rank Adaptation (`LoRALayer`, `LinearWithLoRA`) from scratch in PyTorch, freezing base weights and training low-rank adapter matrices (`~50x` fewer trainable parameters) with zero added inference latency when merged.
