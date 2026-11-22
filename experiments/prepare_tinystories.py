"""Download a prefix of TinyStoriesV2 (GPT-4 generated) and tokenize it with the GPT-2 BPE.

    python experiments/prepare_tinystories.py --train-mb 250

Writes uint16 token files train.bin / val.bin to $NVFP4_DATA (default ~/nvfp4_data). The train file
is fetched with an HTTP Range request so only the first --train-mb megabytes are downloaded.
"""
from __future__ import annotations

import argparse
import os
import urllib.request

import numpy as np
import tiktoken

BASE = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
FILES = {"train": "TinyStoriesV2-GPT4-train.txt", "val": "TinyStoriesV2-GPT4-valid.txt"}


def fetch(url: str, max_bytes: int | None) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "nvfp4-train-sm120"})
    if max_bytes:
        req.add_header("Range", f"bytes=0-{max_bytes - 1}")
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read()
    text = raw.decode("utf-8", errors="ignore")
    if max_bytes:  # drop the trailing partial story
        text = text[: text.rfind("<|endoftext|>")]
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-mb", type=int, default=250)
    ap.add_argument("--out", default=os.environ.get("NVFP4_DATA", os.path.expanduser("~/nvfp4_data")))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    for split, fname in FILES.items():
        path = os.path.join(a.out, f"{split}.bin")
        if os.path.exists(path):
            print(f"{path} exists, skipping")
            continue
        text = fetch(BASE + fname, a.train_mb * 2**20 if split == "train" else None)
        stories = [s.strip() for s in text.split("<|endoftext|>") if s.strip()]
        ids = enc.encode_ordinary_batch(stories, num_threads=8)
        eot = enc.eot_token
        flat = np.fromiter((t for doc in ids for t in (*doc, eot)), dtype=np.uint16)
        flat.tofile(path)
        print(f"{split}: {len(stories)} stories, {flat.size / 1e6:.1f}M tokens -> {path}")


if __name__ == "__main__":
    main()
