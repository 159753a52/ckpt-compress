from transformers import GPT2LMHeadModel
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
print("Downloading gpt2-medium...")
m = GPT2LMHeadModel.from_pretrained("gpt2-medium")
print(f"OK, params: {sum(p.numel() for p in m.parameters())}")
