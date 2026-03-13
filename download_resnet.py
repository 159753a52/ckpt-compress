import torch
import torchvision.models as models
import os

cache_dir = "/lihongliang/fangzl/ckpt-compress/data/models"
os.makedirs(cache_dir, exist_ok=True)

print("Downloading ResNet-50...")
model = models.resnet50(pretrained=True)
save_path = os.path.join(cache_dir, "resnet50_pretrained.pt")
torch.save(model.state_dict(), save_path)
print(f"ResNet-50 saved to {save_path}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
