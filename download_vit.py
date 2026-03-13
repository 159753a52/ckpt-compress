import os
os.environ["HF_HUB_DISABLE_DISK_SPACE_CHECK"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import ViTForImageClassification, ViTFeatureExtractor

cache_dir = "/lihongliang/fangzl/ckpt-compress/data/models"
os.makedirs(cache_dir, exist_ok=True)

print("Downloading ViT-L/32...")
# Try google/vit-large-patch32-224 first
try:
    model = ViTForImageClassification.from_pretrained("google/vit-large-patch32-224", cache_dir=cache_dir)
    print(f"ViT-L/32 downloaded! Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print("Success with google/vit-large-patch32-224")
except Exception as e:
    print(f"Failed with google/vit-large-patch32-224: {e}")
    # Fallback to patch32-384
    try:
        model = ViTForImageClassification.from_pretrained("google/vit-large-patch32-384", cache_dir=cache_dir)
        print(f"ViT-L/32-384 downloaded! Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        print("Success with google/vit-large-patch32-384")
    except Exception as e2:
        print(f"Also failed with 384 variant: {e2}")
        # Try ViT-B/32 as fallback
        model = ViTForImageClassification.from_pretrained("google/vit-base-patch32-224-in21k", cache_dir=cache_dir)
        print(f"ViT-B/32 downloaded as fallback! Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        print("Success with google/vit-base-patch32-224-in21k (fallback)")
