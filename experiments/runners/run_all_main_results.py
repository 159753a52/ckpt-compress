"""
批量运行主结果实验（Table 3）。

从 configs/main_results.yaml 读取配置，对每个 model × sparsity 组合
调用 run_fault_tolerant_training.py，方法以逗号列表形式传入 --methods。

运行示例:
    python experiments/runners/run_all_main_results.py
    python experiments/runners/run_all_main_results.py --only_model gpt2-medium
    python experiments/runners/run_all_main_results.py --dry_run
"""

import sys
import json
import argparse
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent.parent
SCRIPT = ROOT / 'experiments' / 'scripts' / 'run_fault_tolerant_training.py'
CONFIG_FILE = ROOT / 'experiments' / 'configs' / 'main_results.yaml'


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def result_exists(output_dir, ratio):
    """检查该 ratio 的结果文件是否已存在。"""
    result_dir = Path(output_dir)
    if not result_dir.exists():
        return False
    for f in result_dir.glob("*.json"):
        try:
            with open(f, 'r') as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                cfg = data.get('config', {})
                if abs(cfg.get('prune_ratio', -1) - ratio) < 0.001:
                    return True
        except (json.JSONDecodeError, KeyError):
            continue
    return False


def build_command(model_cfg, methods_str, ratio, common_cfg, device):
    """构建单次实验的命令行（一个 model × ratio，内含所有 methods）。"""
    output_dir = f"results/paper_results/main/{model_cfg['name']}"
    cmd = [
        sys.executable, str(SCRIPT),
        '--model', model_cfg['name'],
        '--dataset', model_cfg['dataset'],
        '--methods', methods_str,
        '--prune_ratio', str(ratio),
        '--total_steps', str(common_cfg['total_steps']),
        '--num_recoveries', str(common_cfg.get('num_recoveries', 5)),
        '--lr', str(common_cfg['lr']),
        '--alpha', str(common_cfg['alpha']),
        '--device', device,
        '--output_dir', output_dir,
    ]
    if 'checkpoint_dir' in model_cfg:
        ckpt_dir = Path(model_cfg['checkpoint_dir'])
        # 优先使用最大步数的 checkpoint_step_*.pt 文件（数值排序）
        import re
        ckpt_files = list(ckpt_dir.glob('checkpoint_step_*.pt'))
        if ckpt_files:
            ckpt_files.sort(key=lambda p: int(re.search(r'(\d+)', p.stem).group(1)))
            cmd.extend(['--checkpoint', str(ckpt_files[-1])])
        elif ckpt_dir.is_file():
            cmd.extend(['--checkpoint', str(ckpt_dir)])
        else:
            print(f"  [WARN] 检查点目录 {ckpt_dir} 无 .pt 文件，跳过 --checkpoint")
    if 'batch_size' in model_cfg:
        cmd.extend(['--batch_size', str(model_cfg['batch_size'])])
    if 'seq_length' in model_cfg:
        cmd.extend(['--seq_length', str(model_cfg['seq_length'])])
    return cmd


def main():
    parser = argparse.ArgumentParser(description='批量运行主结果实验 (Table 3)')
    parser.add_argument('--config', type=str, default=str(CONFIG_FILE))
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--only_model', type=str, default=None,
                        help='仅运行指定模型（如 gpt2-medium）')
    parser.add_argument('--only_ratio', type=float, default=None,
                        help='仅运行指定稀疏度')
    parser.add_argument('--dry_run', action='store_true',
                        help='仅打印命令，不执行')
    parser.add_argument('--skip_existing', action='store_true', default=True,
                        help='跳过已完成的实验')
    args = parser.parse_args()

    config = load_config(args.config)
    common = config['common']
    models = config['models']
    ratios = config['sparsity_levels']
    methods = config['methods']

    if args.only_model:
        models = [m for m in models if m['name'] == args.only_model]
    if args.only_ratio is not None:
        ratios = [args.only_ratio]

    # 构造 methods 字符串：importance+allocation 格式
    methods_str = ','.join(m['label'] for m in methods)

    total = len(models) * len(ratios)
    print(f"总运行数: {total}（每次运行包含 {len(methods)} 个方法）")
    print(f"模型: {[m['name'] for m in models]}")
    print(f"稀疏度: {ratios}")
    print(f"方法: {methods_str}")
    print("=" * 70)

    completed, skipped, failed = 0, 0, 0

    for model_cfg in models:
        for ratio in ratios:
            output_dir = f"results/paper_results/main/{model_cfg['name']}"
            tag = f"{model_cfg['name']} | ratio={ratio}"

            if args.skip_existing and result_exists(output_dir, ratio):
                print(f"[SKIP] {tag}")
                skipped += 1
                continue

            cmd = build_command(model_cfg, methods_str, ratio,
                                common, args.device)

            print(f"\n[RUN] {tag}")
            if args.dry_run:
                print(f"  CMD: {' '.join(cmd)}")
                completed += 1
                continue

            try:
                result = subprocess.run(
                    cmd, cwd=str(ROOT), timeout=7200)
                if result.returncode == 0:
                    print(f"  [OK] {tag}")
                    completed += 1
                else:
                    print(f"  [FAIL] {tag} 返回码: {result.returncode}")
                    failed += 1
            except subprocess.TimeoutExpired:
                print(f"  [TIMEOUT] {tag}")
                failed += 1

    print("\n" + "=" * 70)
    print(f"完成: {completed} | 跳过: {skipped} | 失败: {failed}")


if __name__ == '__main__':
    main()
