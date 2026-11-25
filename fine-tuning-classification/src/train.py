"""Training, evaluation, plotting, and inference for GPT classification fine-tuning.

Adapted from Sebastian Raschka, "Build a Large Language Model (From Scratch)"
(Apache License 2.0): https://github.com/rasbt/LLMs-from-scratch
"""

import argparse
from pathlib import Path
import time

import matplotlib.pyplot as plt
import tiktoken
import torch
# Import Dynamo before TensorFlow is loaded by gpt_download to avoid a native
# Triton/TensorFlow initialization crash with recent PyTorch builds.
import torch._dynamo  # noqa: F401
from torch.utils.data import DataLoader

from src.dataset import SpamDataset, prepare_spam_splits
from src.gpt_model import (
    GPT2_CONFIGS,
    GPTModel,
    configure_for_classification,
    load_weights_into_gpt,
)


def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    """Compute classification accuracy using the final causal token logits."""
    model.eval()
    correct_predictions, num_examples = 0, 0
    limit = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= limit:
            break
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        with torch.no_grad():
            logits = model(input_batch)[:, -1, :]
        predicted_labels = torch.argmax(logits, dim=-1)
        num_examples += predicted_labels.shape[0]
        correct_predictions += (predicted_labels == target_batch).sum().item()

    return correct_predictions / num_examples if num_examples > 0 else 0.0


def calc_loss_batch(input_batch, target_batch, model, device):
    """Compute cross-entropy loss on the last token of each sequence."""
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]
    return torch.nn.functional.cross_entropy(logits, target_batch)


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """Compute average cross-entropy loss across batches."""
    if len(data_loader) == 0:
        return float("nan")
    limit = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))
    total_loss = 0.0
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= limit:
            break
        total_loss += calc_loss_batch(input_batch, target_batch, model, device).item()
    return total_loss / limit


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """Evaluate training and validation loss without gradient tracking."""
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def train_classifier_simple(
    model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter
):
    """Train the classification model and track per-step losses and per-epoch accuracies."""
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            examples_seen += input_batch.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(
                    f"Ep {epoch + 1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )

        train_acc = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_acc = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_acc * 100:.2f}% | Validation accuracy: {val_acc * 100:.2f}%")
        train_accs.append(train_acc)
        val_accs.append(val_acc)

    return train_losses, val_losses, train_accs, val_accs, examples_seen


def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    """Classify a single input string as 'spam' or 'not spam'."""
    model.eval()
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]
    effective_max = (
        min(max_length, supported_context_length)
        if max_length is not None
        else supported_context_length
    )

    input_ids = input_ids[:effective_max]
    input_ids += [pad_token_id] * (effective_max - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]
    predicted_label = torch.argmax(logits, dim=-1).item()
    return "spam" if predicted_label == 1 else "not spam"


def plot_values(epochs_seen, examples_seen, train_values, val_values, output_dir, label="loss"):
    """Save a dual-axis training vs. validation plot to PDF."""
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    ax2 = ax1.twiny()
    ax2.plot(examples_seen, train_values, alpha=0)
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()
    out_path = Path(output_dir) / f"{label}-plot.pdf"
    plt.savefig(out_path)
    plt.close(fig)
    return out_path


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Fine-tune a GPT model for SMS spam classification",
    )
    parser.add_argument(
        "--test_mode",
        default=False,
        action="store_true",
        help="Run with a tiny untrained GPTModel on CPU for fast pipeline verification.",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="DataLoader batch size.")
    parser.add_argument("--lr", type=float, default=5e-5, help="AdamW learning rate.")
    parser.add_argument("--data_dir", type=str, default="data", help="Dataset directory.")
    parser.add_argument("--models_dir", type=str, default="gpt2", help="GPT-2 checkpoint dir.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Outputs directory.")
    parser.add_argument("--no_plots", action="store_true", help="Skip saving PDF plots.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_csv, val_csv, test_csv = prepare_spam_splits(args.data_dir)
    tokenizer = tiktoken.get_encoding("gpt2")

    train_dataset = SpamDataset(csv_file=train_csv, max_length=None, tokenizer=tokenizer)
    val_dataset = SpamDataset(
        csv_file=val_csv, max_length=train_dataset.max_length, tokenizer=tokenizer
    )
    test_dataset = SpamDataset(
        csv_file=test_csv, max_length=train_dataset.max_length, tokenizer=tokenizer
    )

    torch.manual_seed(123)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, drop_last=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, drop_last=False)

    if args.test_mode:
        base_config = {
            "vocab_size": 50257,
            "context_length": max(120, train_dataset.max_length),
            "drop_rate": 0.0,
            "qkv_bias": False,
            "emb_dim": 12,
            "n_layers": 1,
            "n_heads": 2,
        }
        model = GPTModel(base_config)
        device = torch.device("cpu")
    else:
        choose_model = "gpt2-small (124M)"
        base_config = {
            "vocab_size": 50257,
            "context_length": 1024,
            "drop_rate": 0.0,
            "qkv_bias": True,
            **GPT2_CONFIGS[choose_model],
        }
        if train_dataset.max_length > base_config["context_length"]:
            raise ValueError(
                f"Dataset length {train_dataset.max_length} exceeds context_length "
                f"{base_config['context_length']}"
            )
        from src.gpt_download import download_and_load_gpt2

        model_size = choose_model.split(" ")[-1].strip("()")
        _, params = download_and_load_gpt2(model_size=model_size, models_dir=args.models_dir)
        model = GPTModel(base_config)
        load_weights_into_gpt(model, params)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(123)
    configure_for_classification(model, emb_dim=base_config["emb_dim"], num_classes=2)
    model.to(device)

    start_time = time.time()
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=args.epochs,
        eval_freq=50,
        eval_iter=5,
    )
    print(f"Training completed in {(time.time() - start_time) / 60:.2f} minutes.")

    train_accuracy = calc_accuracy_loader(train_loader, model, device)
    val_accuracy = calc_accuracy_loader(val_loader, model, device)
    test_accuracy = calc_accuracy_loader(test_loader, model, device)
    print(f"Final Train accuracy:      {train_accuracy * 100:.2f}%")
    print(f"Final Validation accuracy: {val_accuracy * 100:.2f}%")
    print(f"Final Test accuracy:       {test_accuracy * 100:.2f}%")

    checkpoint_path = output_dir / "review_classifier.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved model checkpoint to {checkpoint_path}")

    sample_spam = (
        "You are a winner you have been specially selected to receive $1000 cash or a $2000 award."
    )
    sample_ham = "Hey, just wanted to check if we are still meeting for lunch at 1pm?"
    print(f"Sample 1 -> {classify_review(sample_spam, model, tokenizer, device, train_dataset.max_length)}")
    print(f"Sample 2 -> {classify_review(sample_ham, model, tokenizer, device, train_dataset.max_length)}")

    if not args.no_plots and train_losses:
        epochs_tensor = torch.linspace(0, args.epochs, len(train_losses))
        examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))
        plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses, output_dir)

        epochs_acc_tensor = torch.linspace(0, args.epochs, len(train_accs))
        examples_acc_tensor = torch.linspace(0, examples_seen, len(train_accs))
        plot_values(
            epochs_acc_tensor,
            examples_acc_tensor,
            train_accs,
            val_accs,
            output_dir,
            label="accuracy",
        )


if __name__ == "__main__":
    main()
