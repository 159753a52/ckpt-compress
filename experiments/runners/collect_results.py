"""
聚合实验结果并生成 LaTeX 表格。

扫描 results/paper_results/ 下的 JSON 文件，提取关键指标，
按 model × method × sparsity 聚合生成 Table 3 / Table 4 格式。

运行示例:
    python experiments/runners/collect_results.py
    python experiments/runners/collect_results.py --result_dir results/paper_results/main
    python experiments/runners/collect_results.py --latex
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent.parent


def scan_results(result_dir):
    """扫描目录下所有 JSON 结果文件，返回解析后的记录列表。"""
    records = []
    result_path = Path(result_dir)
    if not result_path.exists():
        print(f"目录不存在: {result_path}")
        return records

    for json_file in sorted(result_path.rglob('*.json')):
        if '_config.json' in json_file.name or 'metadata.json' in json_file.name:
            continue
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        config = data.get('config', {})
        results = data.get('results', [])
        if not results:
            continue

        model = config.get('model', 'unknown')
        dataset = config.get('dataset', 'unknown')
        prune_ratio = config.get('prune_ratio', 0)

        for r in results:
            method = r.get('method', 'unknown')
            records.append({
                'model': model,
                'dataset': dataset,
                'method': method,
                'prune_ratio': prune_ratio,
                'source_file': str(json_file),
                **{k: v for k, v in r.items() if k != 'method'},
            })

    return records


def aggregate_table3(records):
    """按 model × method × sparsity 聚合，打印 Table 3 格式。"""
    # 按 (model, method, ratio) 分组
    grouped = defaultdict(list)
    for r in records:
        key = (r['model'], r['method'], r['prune_ratio'])
        grouped[key].append(r)

    models = sorted({r['model'] for r in records})
    methods = sorted({r['method'] for r in records})
    ratios = sorted({r['prune_ratio'] for r in records})

    print("\n" + "=" * 100)
    print("Table 3: 主要结果")
    print("=" * 100)

    for model in models:
        print(f"\n模型: {model}")
        header = f"{'Method':<35}" + "".join(f"{'ratio=' + str(r):<18}" for r in ratios)
        print(header)
        print("-" * len(header))

        for method in methods:
            row = f"{method:<35}"
            for ratio in ratios:
                key = (model, method, ratio)
                entries = grouped.get(key, [])
                if entries:
                    # 取最后一条（最终 loss/accuracy）
                    last = entries[-1] if isinstance(entries, list) else entries
                    # 尝试提取常见 metric
                    val, _ = _extract_metric(last)
                    row += f"{val:<18}"
                else:
                    row += f"{'—':<18}"
            print(row)

    print("=" * 100)


def _extract_metric(record):
    """从记录中提取主要指标值和对应的 key。"""
    for key in ['accuracy', 'top1_accuracy', 'perplexity',
                'final_loss', 'val_loss', 'loss', 'metric_value']:
        if key in record:
            val = record[key]
            if isinstance(val, float):
                return f"{val:.4f}", key
            return str(val), key
    # 取 metrics 字典
    metrics = record.get('metrics', {})
    if isinstance(metrics, dict) and metrics:
        first_key = next(iter(metrics))
        val = metrics[first_key]
        if isinstance(val, float):
            return f"{val:.4f}", first_key
        return str(val), first_key
    return "—", None


def generate_latex(records, ratios=None):
    """生成 LaTeX 表格代码。"""
    models = sorted({r['model'] for r in records})
    methods = sorted({r['method'] for r in records})
    if ratios is None:
        ratios = sorted({r['prune_ratio'] for r in records})

    grouped = defaultdict(list)
    for r in records:
        key = (r['model'], r['method'], r['prune_ratio'])
        grouped[key].append(r)

    n_ratios = len(ratios)
    col_spec = "l" + "c" * n_ratios

    print("\\begin{table}[t]")
    print("\\centering")
    print("\\small")

    for model in models:
        print(f"\\caption{{Results on {model}}}")
        print(f"\\begin{{tabular}}{{{col_spec}}}")
        print("\\toprule")
        header = "Method & " + " & ".join(f"{r:.0%}" for r in ratios) + " \\\\"
        print(header)
        print("\\midrule")

        best_per_ratio = {}
        for ratio in ratios:
            best_val = None
            for method in methods:
                entries = grouped.get((model, method, ratio), [])
                if entries:
                    val_str, metric_key = _extract_metric(entries[-1])
                    try:
                        val = float(val_str)
                        is_higher_better = metric_key in ('accuracy', 'top1_accuracy')
                        if best_val is None:
                            best_val = val
                            best_per_ratio[ratio] = method
                        elif is_higher_better and val > best_val:
                            best_val = val
                            best_per_ratio[ratio] = method
                        elif not is_higher_better and val < best_val:
                            best_val = val
                            best_per_ratio[ratio] = method
                    except ValueError:
                        pass

        for method in methods:
            cells = [method.replace('_', '\\_')]
            for ratio in ratios:
                entries = grouped.get((model, method, ratio), [])
                if entries:
                    val_str, _ = _extract_metric(entries[-1])
                    if best_per_ratio.get(ratio) == method:
                        cells.append(f"\\textbf{{{val_str}}}")
                    else:
                        cells.append(val_str)
                else:
                    cells.append("—")
            print(" & ".join(cells) + " \\\\")

        print("\\bottomrule")
        print("\\end{tabular}")
        print()

    print("\\end{table}")


def main():
    parser = argparse.ArgumentParser(description='聚合实验结果')
    parser.add_argument('--result_dir', type=str,
                        default=str(ROOT / 'results' / 'paper_results'),
                        help='结果目录')
    parser.add_argument('--latex', action='store_true',
                        help='输出 LaTeX 表格')
    args = parser.parse_args()

    records = scan_results(args.result_dir)
    if not records:
        print("未找到实验结果")
        return

    print(f"扫描到 {len(records)} 条记录")
    aggregate_table3(records)

    if args.latex:
        print("\n" + "=" * 100)
        print("LaTeX 输出")
        print("=" * 100)
        generate_latex(records)


if __name__ == '__main__':
    main()
