"""
模型权重下载脚本。

提供下载 GPT-2 Small、GPT-2 Medium、BERT-Base、BERT-Large 和 ResNet-50 预训练模型的功能。
支持 HuggingFace、镜像源和 ModelScope。
"""

import os
import argparse
from typing import List, Optional, Dict


def download_gpt2_small(cache_dir: str = "./data/models") -> bool:
    """
    下载 GPT-2 Small 预训练权重。
    
    参数:
        cache_dir: 模型缓存目录
        
    返回:
        bool: 下载成功返回 True，失败返回 False
    """
    try:
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
        
        # 导入 transformers
        from transformers import GPT2LMHeadModel
        
        # 下载模型
        print(f"Downloading GPT-2 Small to {cache_dir}...")
        GPT2LMHeadModel.from_pretrained('gpt2', cache_dir=cache_dir)
        
        print("✓ GPT-2 Small downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Failed to download GPT-2 Small: {e}")
        return False


def download_gpt2_medium(cache_dir: str = "./data/models") -> bool:
    """
    下载 GPT-2 Medium 预训练权重。
    
    参数:
        cache_dir: 模型缓存目录
        
    返回:
        bool: 下载成功返回 True，失败返回 False
    """
    try:
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
        
        # 导入 transformers
        from transformers import GPT2LMHeadModel
        
        # 下载模型
        print(f"Downloading GPT-2 Medium to {cache_dir}...")
        GPT2LMHeadModel.from_pretrained('gpt2-medium', cache_dir=cache_dir)
        
        print("✓ GPT-2 Medium downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Failed to download GPT-2 Medium: {e}")
        return False


def download_resnet50_imagenet(cache_dir: str = "./data/models") -> bool:
    """
    下载 ResNet-50 ImageNet 预训练权重。

    参数:
        cache_dir: 模型缓存目录

    返回:
        bool: 下载成功返回 True，失败返回 False
    """
    try:
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)

        # 设置 torch hub 缓存目录
        import torch
        torch.hub.set_dir(cache_dir)

        # 导入 torchvision
        from torchvision import models

        # 下载模型
        print(f"Downloading ResNet-50 to {cache_dir}...")
        models.resnet50(pretrained=True)

        print("✓ ResNet-50 downloaded successfully!")
        return True

    except Exception as e:
        print(f"✗ Failed to download ResNet-50: {e}")
        return False


def download_bert_base(cache_dir: str = "./data/models") -> bool:
    """
    下载 BERT-Base 预训练权重。

    参数:
        cache_dir: 模型缓存目录

    返回:
        bool: 下载成功返回 True，失败返回 False
    """
    try:
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)

        # 导入 transformers
        from transformers import BertForMaskedLM

        # 下载模型
        print(f"Downloading BERT-Base to {cache_dir}...")
        BertForMaskedLM.from_pretrained('bert-base-uncased', cache_dir=cache_dir)

        print("✓ BERT-Base downloaded successfully!")
        return True

    except Exception as e:
        print(f"✗ Failed to download BERT-Base: {e}")
        return False


def download_bert_large(cache_dir: str = "./data/models") -> bool:
    """
    下载 BERT-Large 预训练权重。

    参数:
        cache_dir: 模型缓存目录

    返回:
        bool: 下载成功返回 True，失败返回 False
    """
    try:
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)

        # 导入 transformers
        from transformers import BertForMaskedLM

        # 下载模型
        print(f"Downloading BERT-Large to {cache_dir}...")
        print("Note: BERT-Large is ~1.3GB, this may take a while...")
        BertForMaskedLM.from_pretrained('bert-large-uncased', cache_dir=cache_dir)

        print("✓ BERT-Large downloaded successfully!")
        return True

    except Exception as e:
        print(f"✗ Failed to download BERT-Large: {e}")
        return False


def download_all_models(
    cache_dir: str = "./data/models",
    models: Optional[List[str]] = None
) -> Dict[str, bool]:
    """
    下载所有或指定的模型。

    参数:
        cache_dir: 模型缓存目录
        models: 要下载的模型列表，None 表示下载所有

    返回:
        Dict[str, bool]: 每个模型的下载结果
    """
    # 可用的模型
    available_models = {
        'gpt2-small': download_gpt2_small,
        'gpt2-medium': download_gpt2_medium,
        'bert-base': download_bert_base,
        'bert-large': download_bert_large,
        'resnet50': download_resnet50_imagenet,
    }

    # 如果未指定模型，下载所有
    if models is None:
        models = list(available_models.keys())

    # 下载每个模型
    results = {}
    for model_name in models:
        if model_name in available_models:
            print(f"\n{'='*60}")
            print(f"Downloading {model_name}...")
            print(f"{'='*60}")
            results[model_name] = available_models[model_name](cache_dir)
        else:
            print(f"✗ Unknown model: {model_name}")
            results[model_name] = False

    # 打印总结
    print(f"\n{'='*60}")
    print("Download Summary:")
    print(f"{'='*60}")
    for model_name, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {model_name}: {'Success' if success else 'Failed'}")

    return results


def main():
    """命令行接口。"""
    parser = argparse.ArgumentParser(
        description="Download pretrained models for ckpt-compress project"
    )

    parser.add_argument(
        '--model',
        type=str,
        help='Model to download (gpt2-small, gpt2-medium, bert-base, bert-large, resnet50). '
             'Can specify multiple models separated by commas.'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Download all models'
    )

    parser.add_argument(
        '--cache_dir',
        type=str,
        default='./data/models',
        help='Directory to cache models (default: ./data/models)'
    )

    args = parser.parse_args()

    # 下载模型
    if args.all:
        download_all_models(cache_dir=args.cache_dir, models=None)
    elif args.model:
        models = [m.strip() for m in args.model.split(',')]
        download_all_models(cache_dir=args.cache_dir, models=models)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
