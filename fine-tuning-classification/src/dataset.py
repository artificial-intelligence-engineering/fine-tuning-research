"""Dataset download, balancing, splitting, and PyTorch Dataset for SMS spam classification.

Adapted from Sebastian Raschka, "Build a Large Language Model (From Scratch)"
(Apache License 2.0): https://github.com/rasbt/LLMs-from-scratch
"""

import os
from pathlib import Path
import zipfile

import pandas as pd
import requests
import torch
from torch.utils.data import Dataset


PRIMARY_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
BACKUP_URL = "https://f001.backblazeb2.com/file/LLMs-from-scratch/sms%2Bspam%2Bcollection.zip"


def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    """Download and extract the UCI SMS Spam Collection dataset."""
    data_file_path = Path(data_file_path)
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return

    zip_path = Path(zip_path)
    extracted_path = Path(extracted_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    extracted_path.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with open(zip_path, "wb") as out_file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                out_file.write(chunk)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    original_file_path = extracted_path / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")


def create_balanced_dataset(df):
    """Downsample 'ham' messages to match the count of 'spam' messages."""
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    return pd.concat([ham_subset, df[df["Label"] == "spam"]]).reset_index(drop=True)


def random_split(df, train_frac, validation_frac):
    """Shuffle and split a DataFrame into train, validation, and test DataFrames."""
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)
    return df[:train_end], df[train_end:validation_end], df[validation_end:]


def prepare_spam_splits(data_dir):
    """Download, balance, split, and persist train/validation/test CSV files."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    zip_path = data_dir / "sms_spam_collection.zip"
    extracted_path = data_dir / "sms_spam_collection"
    data_file_path = extracted_path / "SMSSpamCollection.tsv"

    try:
        download_and_unzip_spam_data(PRIMARY_URL, zip_path, extracted_path, data_file_path)
    except (requests.exceptions.RequestException, TimeoutError) as exc:
        print(f"Primary URL failed: {exc}. Trying backup URL...")
        download_and_unzip_spam_data(BACKUP_URL, zip_path, extracted_path, data_file_path)

    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
    balanced_df = create_balanced_dataset(df)
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
    train_csv = data_dir / "train.csv"
    val_csv = data_dir / "validation.csv"
    test_csv = data_dir / "test.csv"

    train_df.to_csv(train_csv, index=False)
    validation_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    return train_csv, val_csv, test_csv


class SpamDataset(Dataset):
    """PyTorch Dataset that tokenizes, truncates, and pads SMS messages."""

    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]

        if max_length is None:
            self.max_length = max(len(t) for t in self.encoded_texts) if self.encoded_texts else 0
        else:
            self.max_length = max_length
            self.encoded_texts = [t[: self.max_length] for t in self.encoded_texts]

        self.encoded_texts = [
            t + [pad_token_id] * (self.max_length - len(t))
            for t in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = int(self.data.iloc[index]["Label"])
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )

    def __len__(self):
        return len(self.data)
