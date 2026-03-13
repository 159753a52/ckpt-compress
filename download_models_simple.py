import os
import torch
import torchvision.models as models

# Disable disk space check
os.environ["HF_HUB_DISABLE_DISK_SPACE_CHECK"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

cache_dir = "/lihongliang/fangzl/ckpt-compress/data/models"
os.makedirs(cache_dir, exist_ok=True)

print("=" * 60)
print("Downloading ResNet-50...")
print("=" * 60)
try:
    resnet50 = models.resnet50(pretrained=True)
    save_path = os.path.join(cache_dir, "resnet50_pretrained.pt")
    torch.save(resnet50.state_dict(), save_path)
    print(f"✓ ResNet-50 saved to {save_path}")
    print(f"  Parameters: {sum(p.numel() for p in resnet50.parameters()) / 1e6:.1f}M")
    del resnet50
except Exception as e:
    print(f"✗ ResNet-50 download failed: {e}")

print("\n" + "=" * 60)
print("Downloading ViT models...")
print("=" * 60)

# Try ViT from torchvision first (simpler, no HuggingFace dependency)
try:
    from torchvision.models import vit_b_32, ViT_B_32_Weights
    print("Trying torchvision ViT-B/32...")
    vit = vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1)
    save_path = os.path.join(cache_dir, "vit_b_32_pretrained.pt")
    torch.save(vit.state_dict(), save_path)
    print(f"✓ ViT-B/32 saved to {save_path}")
    print(f"  Parameters: {sum(p.numel() for p in vit.parameters()) / 1e6:.1f}M")
    del vit
except Exception as e:
    print(f"✗ torchvision ViT failed: {e}")

# Try HuggingFace ViT
try:
    from transformers import ViTForImageClassification
    print("\nTrying HuggingFace ViT-B/16...")
    # Use ViT-B/16 which is more common and reliable
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        cache_dir=cache_dir
    )
    print(f"✓ ViT-B/16 downloaded to {cache_dir}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    del model
except Exception as e:
    print(f"✗ HuggingFace ViT-B/16 failed: {e}")

print("\n" + "=" * 60)
print("Download Summary")
print("=" * 60)
print(f"Models directory: {cache_dir}")
print("Files created:")
for f in os.listdir(cache_dir):
    fpath = os.path.join(cache_dir, f)
    if os.path.isfile(fpath):
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"  - {f} ({size_mb:.1f} MB)")
