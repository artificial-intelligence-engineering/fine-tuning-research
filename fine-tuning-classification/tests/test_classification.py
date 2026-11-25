"""Unit and smoke tests for GPT classification fine-tuning."""

import pandas as pd
import tiktoken
import torch
from torch.utils.data import DataLoader

from src.dataset import SpamDataset, create_balanced_dataset, random_split
from src.gpt_model import GPTModel, configure_for_classification
from src.train import calc_accuracy_loader, calc_loss_batch, classify_review, train_classifier_simple


TINY_CFG = {
    "vocab_size": 50257,
    "context_length": 32,
    "drop_rate": 0.0,
    "qkv_bias": False,
    "emb_dim": 16,
    "n_layers": 2,
    "n_heads": 2,
}


def test_balance_and_split():
    rows = [{"Label": "ham", "Text": f"ham {i}"} for i in range(10)]
    rows += [{"Label": "spam", "Text": f"spam {i}"} for i in range(4)]
    df = pd.DataFrame(rows)

    balanced = create_balanced_dataset(df)
    assert len(balanced) == 8
    assert (balanced["Label"] == "spam").sum() == 4
    assert (balanced["Label"] == "ham").sum() == 4

    train_df, val_df, test_df = random_split(balanced, 0.5, 0.25)
    assert len(train_df) == 4
    assert len(val_df) == 2
    assert len(test_df) == 2


def test_spam_dataset_padding_and_truncation(tmp_path):
    csv_path = tmp_path / "sample.csv"
    pd.DataFrame(
        [
            {"Label": 0, "Text": "short"},
            {"Label": 1, "Text": "this is a much longer message for testing padding"},
        ]
    ).to_csv(csv_path, index=False)

    tokenizer = tiktoken.get_encoding("gpt2")
    ds = SpamDataset(csv_path, tokenizer)
    assert len(ds) == 2
    x0, y0 = ds[0]
    x1, y1 = ds[1]
    assert x0.shape == x1.shape == (ds.max_length,)
    assert y0.item() == 0
    assert y1.item() == 1
    assert x0[-1].item() == 50256

    ds_trunc = SpamDataset(csv_path, tokenizer, max_length=3)
    assert ds_trunc[1][0].shape == (3,)


def test_configure_for_classification_freezing_and_head():
    torch.manual_seed(0)
    model = GPTModel(TINY_CFG)
    configure_for_classification(model, emb_dim=TINY_CFG["emb_dim"], num_classes=2)

    assert model.out_head.out_features == 2
    assert all(not p.requires_grad for p in model.tok_emb.parameters())
    assert all(not p.requires_grad for p in model.trf_blocks[0].parameters())
    assert all(p.requires_grad for p in model.trf_blocks[-1].parameters())
    assert all(p.requires_grad for p in model.final_norm.parameters())
    assert all(p.requires_grad for p in model.out_head.parameters())


def test_training_step_and_inference(tmp_path):
    csv_path = tmp_path / "train.csv"
    pd.DataFrame(
        [
            {"Label": 0, "Text": "meeting at noon"},
            {"Label": 1, "Text": "win free cash prize now"},
            {"Label": 0, "Text": "see you tomorrow"},
            {"Label": 1, "Text": "urgent claim your reward"},
        ]
    ).to_csv(csv_path, index=False)

    tokenizer = tiktoken.get_encoding("gpt2")
    ds = SpamDataset(csv_path, tokenizer, max_length=16)
    loader = DataLoader(ds, batch_size=2, shuffle=False)

    torch.manual_seed(123)
    model = GPTModel(TINY_CFG)
    configure_for_classification(model, emb_dim=TINY_CFG["emb_dim"], num_classes=2)
    device = torch.device("cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x, y = next(iter(loader))
    loss = calc_loss_batch(x, y, model, device)
    assert loss.item() > 0

    train_losses, _, _, _, seen = train_classifier_simple(
        model, loader, loader, optimizer, device, num_epochs=1, eval_freq=1, eval_iter=1
    )
    assert seen == 4
    assert len(train_losses) >= 1

    acc = calc_accuracy_loader(loader, model, device)
    assert 0.0 <= acc <= 1.0

    pred = classify_review("free cash prize", model, tokenizer, device, max_length=16)
    assert pred in ("spam", "not spam")
