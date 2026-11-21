#!/usr/bin/env python
"""Download (through the lab proxy) and tokenize the data of the A30 long runs.

* Alpaca (full 52,002 samples): data/alpaca/train.parquet
* FineWeb-Edu sample-10BT shards: tokenized with the Pythia (GPT-NeoX) tokenizer,
  documents separated by the end-of-text token 0, written as uint16 streams
  data/fineweb_edu/train.bin and val.bin (the last --val-tokens tokens).

Downloads resume after interruptions and are accepted only at the size reported by
the mirror's API.  Every finished stage leaves a marker file, so the script can be
rerun until data/fineweb_edu/DONE exists.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

MIRROR = "https://hf-mirror.com"
ALPACA = ("tatsu-lab/alpaca", "data/train-00000-of-00001-a09b74b3ef9c3b56.parquet")
FINEWEB = "HuggingFaceFW/fineweb-edu"


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def api_size(repo, path):
    url = f"{MIRROR}/api/datasets/{repo}/tree/main/{os.path.dirname(path)}"
    out = subprocess.run(["curl", "-s", "--max-time", "60", url], capture_output=True, text=True, check=True).stdout
    for f in json.loads(out):
        if f.get("path") == path:
            return int(f["size"])
    raise RuntimeError(f"{path} not listed by the mirror")


def download(repo, path, dest: Path):
    if dest.exists():
        return
    size = api_size(repo, path)
    part = dest.with_suffix(dest.suffix + ".part")
    url = f"{MIRROR}/datasets/{repo}/resolve/main/{path}"
    for attempt in range(1, 200):
        have = part.stat().st_size if part.exists() else 0
        if have == size:
            break
        if have > size:
            part.unlink()
            continue
        log(f"download {path}: {have / 1e9:.2f}/{size / 1e9:.2f} GB (attempt {attempt})")
        subprocess.run(["curl", "-sSL", "-C", "-", "--retry", "5", "--retry-delay", "10",
                        "--speed-limit", "20000", "--speed-time", "120", "-o", str(part), url])
        time.sleep(5)
    if part.stat().st_size != size:
        raise RuntimeError(f"{path}: size {part.stat().st_size} != {size}")
    os.replace(part, dest)
    log(f"downloaded {path} ({size / 1e9:.2f} GB)")


_TOK = None


def _init(tok_name):
    global _TOK
    from transformers import AutoTokenizer

    _TOK = AutoTokenizer.from_pretrained(tok_name)


def _encode(texts):
    ids = _TOK(texts, add_special_tokens=False)["input_ids"]
    out = []
    for x in ids:
        out.extend(x)
        out.append(0)
    arr = np.asarray(out, dtype=np.uint32)
    if arr.size and arr.max() >= 65535:
        raise ValueError("token id does not fit in uint16")
    return arr.astype(np.uint16)


def tokenize(shards, out_dir: Path, tok_name, workers, val_tokens):
    import pyarrow.parquet as pq

    stream = out_dir / "all.bin.tmp"
    total = 0
    with stream.open("wb") as fh, mp.Pool(workers, initializer=_init, initargs=(tok_name,)) as pool:
        for shard in shards:
            pf = pq.ParquetFile(shard)

            def batches():
                for b in pf.iter_batches(batch_size=1000, columns=["text"]):
                    yield b.column(0).to_pylist()

            t0 = time.time()
            for arr in pool.imap(_encode, batches(), chunksize=1):
                fh.write(arr.tobytes())
                total += arr.size
            log(f"tokenized {shard.name}: running total {total / 1e9:.3f} B tokens ({time.time() - t0:.0f} s)")
    data = np.memmap(stream, dtype=np.uint16, mode="r")
    n_train = len(data) - val_tokens
    np.asarray(data[n_train:]).tofile(out_dir / "val.bin")
    del data
    with stream.open("r+b") as fh:
        fh.truncate(n_train * 2)
    os.replace(stream, out_dir / "train.bin")
    return n_train


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--shards", default="000_00000,001_00000")
    ap.add_argument("--tokenizer", default="EleutherAI/pythia-410m")
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--val-tokens", type=int, default=20_000_000)
    args = ap.parse_args()

    alp = args.root / "alpaca" / "train.parquet"
    alp.parent.mkdir(parents=True, exist_ok=True)
    download(*ALPACA, alp)
    (args.root / "alpaca" / "DONE").write_text("ok\n")

    fw = args.root / "fineweb_edu"
    raw = fw / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    if (fw / "DONE").exists():
        log("fineweb_edu already prepared")
        return
    shards = []
    for s in args.shards.split(","):
        dest = raw / f"{s}.parquet"
        download(FINEWEB, f"sample/10BT/{s}.parquet", dest)
        shards.append(dest)
    n_train = tokenize(shards, fw, args.tokenizer, args.workers, args.val_tokens)
    info = {"shards": [s.name for s in shards], "train_tokens": int(n_train), "val_tokens": args.val_tokens,
            "tokenizer": args.tokenizer, "eos_id": 0, "dtype": "uint16"}
    (fw / "info.json").write_text(json.dumps(info, indent=1))
    (fw / "DONE").write_text("ok\n")
    log(f"fineweb_edu ready: {n_train / 1e9:.3f} B train tokens")


if __name__ == "__main__":
    sys.exit(main())
