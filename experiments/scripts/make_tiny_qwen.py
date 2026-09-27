"""Create a randomly initialized 3-layer Qwen2 with a real tokenizer for smoke tests.

Only used to exercise the orchestration of run_probe_coverage_32b.py; its
numbers carry no experimental meaning.
"""
import argparse

import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=len(tok),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        tie_word_embeddings=False,
        torch_dtype="bfloat16",
    )
    model = Qwen2ForCausalLM(cfg).to(torch.bfloat16)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)


if __name__ == "__main__":
    main()
