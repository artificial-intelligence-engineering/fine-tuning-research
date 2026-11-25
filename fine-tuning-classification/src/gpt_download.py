"""Download and parse OpenAI GPT-2 TensorFlow checkpoints.

Adapted from Sebastian Raschka, "Build a Large Language Model (From Scratch)"
(Apache License 2.0): https://github.com/rasbt/LLMs-from-scratch
"""

import json
import os

import numpy as np
import requests
import tensorflow as tf
from tqdm import tqdm


def download_and_load_gpt2(model_size, models_dir):
    """Download OpenAI GPT-2 checkpoint files and return (settings, params)."""
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size must be one of {allowed_sizes}")

    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    backup_base_url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/gpt2"
    filenames = [
        "checkpoint",
        "encoder.json",
        "hparams.json",
        "model.ckpt.data-00000-of-00001",
        "model.ckpt.index",
        "model.ckpt.meta",
        "vocab.bpe",
    ]

    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = f"{base_url}/{model_size}/{filename}"
        backup_url = f"{backup_base_url}/{model_size}/{filename}"
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path, backup_url)

    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    with open(os.path.join(model_dir, "hparams.json"), "r", encoding="utf-8") as f:
        settings = json.load(f)
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params


def download_file(url, destination, backup_url=None):
    """Stream-download a file with size check and optional fallback URL."""

    def _attempt_download(download_url):
        response = requests.get(download_url, stream=True, timeout=60)
        response.raise_for_status()
        file_size = int(response.headers.get("Content-Length", 0))

        if os.path.exists(destination):
            file_size_local = os.path.getsize(destination)
            if file_size and file_size == file_size_local:
                print(f"File already exists and is up-to-date: {destination}")
                return True

        block_size = 1024
        desc = os.path.basename(download_url)
        with tqdm(total=file_size, unit="iB", unit_scale=True, desc=desc) as progress_bar:
            with open(destination, "wb") as file:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        file.write(chunk)
                        progress_bar.update(len(chunk))
        return True

    try:
        if _attempt_download(url):
            return
    except requests.exceptions.RequestException:
        if backup_url is not None:
            print(f"Primary URL ({url}) failed. Attempting backup URL: {backup_url}")
            _attempt_download(backup_url)
            return
        raise


def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    """Extract weights from a TensorFlow GPT-2 checkpoint into nested dicts."""
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    for name, _ in tf.train.list_variables(ckpt_path):
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))
        variable_name_parts = name.split("/")[1:]  # Skip 'model/' prefix

        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        target_dict[variable_name_parts[-1]] = variable_array

    return params
