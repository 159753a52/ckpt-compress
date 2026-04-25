"""Download bert-base-uncased tokenizer and GLUE datasets for offline use."""
import os
os.environ.pop('TRANSFORMERS_OFFLINE', None)
os.environ.pop('HF_HUB_OFFLINE', None)
os.environ.pop('HF_DATASETS_OFFLINE', None)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('bert-base-uncased')
print(f"Tokenizer OK: {t.name_or_path}")

from datasets import load_dataset
for name in ['sst2', 'mnli']:
    ds = load_dataset('glue', name)
    print(f"GLUE {name} OK: {ds}")
