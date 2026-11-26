"""分析 fit_results.json，为可视化方案提供决策依据"""
import json
import numpy as np

def analyze(path, label):
    with open(path, 'r') as f:
        data = json.load(f)
    
    print(f"\n{'='*60}")
    print(f"  {label}  ({len(data)} layers)")
    print(f"{'='*60}")
    
    dists = ['weibull', 'gamma', 'lognormal', 'exponential']
    ks_vals = {d: [] for d in dists}
    aic_vals = {d: [] for d in dists}
    
    for layer in data:
        for d in dists:
            ks_vals[d].append(layer[f'{d}_ks'])
            aic_vals[d].append(layer[f'{d}_aic'])
    
    # 1. KS-D 统计
    print("\n--- KS-D 统计 (越小越好) ---")
    print(f"{'分布':<12} {'mean':>10} {'std':>10} {'min':>10} {'max':>10} {'median':>10}")
    for d in dists:
        arr = np.array(ks_vals[d])
        print(f"{d:<12} {arr.mean():>10.4f} {arr.std():>10.4f} {arr.min():>10.4f} {arr.max():>10.4f} {np.median(arr):>10.4f}")
    
    # 2. 每层最优分布统计
    print("\n--- 每层最优分布 (KS-D最低) ---")
    best_count = {d: 0 for d in dists}
    for i, layer in enumerate(data):
        best = min(dists, key=lambda d: layer[f'{d}_ks'])
        best_count[best] += 1
    for d in dists:
        print(f"  {d}: {best_count[d]} / {len(data)} layers ({best_count[d]/len(data)*100:.1f}%)")
    
    # 3. AIC 统计
    print("\n--- AIC 统计 (越小越好) ---")
    print(f"{'分布':<12} {'mean':>14} {'std':>14}")
    for d in dists:
        arr = np.array(aic_vals[d])
        print(f"{d:<12} {arr.mean():>14.1f} {arr.std():>14.1f}")
    
    # 4. 每层最优分布 (AIC)
    print("\n--- 每层最优分布 (AIC最低) ---")
    best_aic = {d: 0 for d in dists}
    for layer in data:
        best = min(dists, key=lambda d: layer[f'{d}_aic'])
        best_aic[best] += 1
    for d in dists:
        print(f"  {d}: {best_aic[d]} / {len(data)} layers ({best_aic[d]/len(data)*100:.1f}%)")
    
    # 5. Weibull shape 参数分布
    print("\n--- Weibull shape 参数 (c) ---")
    shapes = []
    for layer in data:
        # params format: "(c, loc, scale)"
        params_str = layer['weibull_params']
        c = float(params_str.strip('()').split(',')[0])
        shapes.append(c)
    shapes = np.array(shapes)
    print(f"  mean={shapes.mean():.4f}, std={shapes.std():.4f}, min={shapes.min():.4f}, max={shapes.max():.4f}")
    print(f"  c<1 (重尾): {(shapes<1).sum()}/{len(shapes)} ({(shapes<1).mean()*100:.1f}%)")
    
    # 6. 逐层 KS-D 值一览（Weibull vs Gamma）
    print("\n--- 逐层 KS-D: Weibull vs Gamma (前10层 + 后5层) ---")
    print(f"{'Layer':<55} {'Weibull':>8} {'Gamma':>8} {'LogN':>8} {'Exp':>8}")
    for layer in data[:10]:
        name = layer['layer'].replace('transformer.h.', 'h.').replace('bert.encoder.layer.', 'L')
        print(f"{name:<55} {layer['weibull_ks']:>8.4f} {layer['gamma_ks']:>8.4f} {layer['lognormal_ks']:>8.4f} {layer['exponential_ks']:>8.4f}")
    print("  ...")
    for layer in data[-5:]:
        name = layer['layer'].replace('transformer.h.', 'h.').replace('bert.encoder.layer.', 'L')
        print(f"{name:<55} {layer['weibull_ks']:>8.4f} {layer['gamma_ks']:>8.4f} {layer['lognormal_ks']:>8.4f} {layer['exponential_ks']:>8.4f}")
    
    return ks_vals, aic_vals

# 运行分析
ks1, aic1 = analyze(r'gpt2m_fit_results.json', 'GPT-2 Medium + WikiText-2')
ks2, aic2 = analyze(r'bert_fit_results.json', 'BERT-Large + SST-2')

# 7. 综合比较
print(f"\n{'='*60}")
print("  综合：两模型合并看 Weibull 优势")
print(f"{'='*60}")
dists = ['weibull', 'gamma', 'lognormal', 'exponential']
for d in dists:
    all_ks = np.array(ks1[d] + ks2[d])
    print(f"  {d:<12} mean KS-D = {all_ks.mean():.4f} ± {all_ks.std():.4f}")
