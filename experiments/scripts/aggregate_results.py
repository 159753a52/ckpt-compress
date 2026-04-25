"""
Aggregate all FT experiment results from results/paper_results/ directory.
Produces a clean summary table for each model/dataset combo.
"""

import json
import os
from pathlib import Path
from collections import defaultdict


def load_all_results(results_dir):
    """Load all JSON result files from the results directory."""
    results = []
    for dirpath, _, filenames in os.walk(results_dir):
        for fn in filenames:
            if fn.endswith('.json') and not fn.endswith('_config.json'):
                fp = os.path.join(dirpath, fn)
                with open(fp) as f:
                    data = json.load(f)
                if 'summary' in data:
                    data['_file'] = fp
                    data['_dir'] = os.path.basename(dirpath)
                    results.append(data)
    return results


def extract_final_metrics(result):
    """Extract final metrics per method from a result JSON."""
    summary = result.get('summary', {})
    finals = summary.get('final', {})
    config = summary.get('config', result.get('config', {}))
    
    model = config.get('model', '?')
    dataset = config.get('dataset', '?')
    prune_ratio = config.get('prune_ratio', 0.3)
    K = config.get('num_recoveries', 5)
    lr = config.get('lr', '?')
    
    out = []
    for method, metrics in finals.items():
        entry = {
            'model': model,
            'dataset': dataset,
            'prune_ratio': prune_ratio,
            'K': K,
            'lr': lr,
            'method': method,
        }
        if 'accuracy' in metrics:
            entry['metric'] = 'accuracy'
            entry['value'] = metrics['accuracy']
        elif 'perplexity' in metrics:
            entry['metric'] = 'perplexity'
            entry['value'] = metrics['perplexity']
        elif 'pearson' in metrics:
            entry['metric'] = 'pearson'
            entry['value'] = metrics['pearson']
        else:
            entry['metric'] = 'loss'
            entry['value'] = metrics.get('loss', 0)
        out.append(entry)
    return out


def main():
    results_dir = Path(__file__).parent.parent.parent / 'results' / 'paper_results'
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return
    
    all_results = load_all_results(results_dir)
    print(f"Found {len(all_results)} result files\n")
    
    # Flatten to per-method entries
    entries = []
    for r in all_results:
        entries.extend(extract_final_metrics(r))
    
    if not entries:
        print("No results found.")
        return
    
    # Group by (model, dataset, K)
    groups = defaultdict(list)
    for e in entries:
        key = (e['model'], e['dataset'], e['K'])
        groups[key].append(e)
    
    # Print summary tables
    for (model, dataset, K), group in sorted(groups.items()):
        metric_name = group[0]['metric']
        higher_better = metric_name in ('accuracy', 'pearson')
        
        print(f"{'='*60}")
        print(f"{model} / {dataset} (K={K})")
        print(f"Metric: {metric_name} {'(higher=better)' if higher_better else '(lower=better)'}")
        print(f"{'='*60}")
        
        # Group by prune_ratio
        by_ratio = defaultdict(dict)
        for e in group:
            by_ratio[e['prune_ratio']][e['method']] = e['value']
        
        # Get all methods
        all_methods = sorted(set(e['method'] for e in group))
        
        # Print header
        ratios = sorted(by_ratio.keys())
        header = f"{'Method':<30}"
        for r in ratios:
            header += f" | {r:>8.0%}"
        print(header)
        print('-' * len(header))
        
        # Print rows
        for method in all_methods:
            row = f"{method:<30}"
            for r in ratios:
                val = by_ratio[r].get(method)
                if val is not None:
                    if metric_name == 'perplexity':
                        row += f" | {val:>8.2f}"
                    elif metric_name in ('accuracy', 'pearson'):
                        row += f" | {val:>7.1%}"
                    else:
                        row += f" | {val:>8.4f}"
                else:
                    row += f" | {'—':>8}"
            print(row)
        
        # Mark best per ratio
        print()
        best_row = f"{'BEST':<30}"
        for r in ratios:
            vals = by_ratio[r]
            if vals:
                if higher_better:
                    best_method = max(vals, key=vals.get)
                else:
                    best_method = min(vals, key=vals.get)
                best_row += f" | {best_method[:8]:>8}"
            else:
                best_row += f" | {'?':>8}"
        print(best_row)
        print()


if __name__ == '__main__':
    main()
