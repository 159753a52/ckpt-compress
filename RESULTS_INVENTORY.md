# 实验结果清单

**生成时间**: 2026-02-26  
**总目录数**: 36  
**总存储**: 64MB  
**可视化**: 138 PNG, 51 CSV

---

## 快速索引

### ⭐⭐⭐ 论文核心结果 (8个)

| # | 实验名称 | 价值 | 完整度 | 大小 | 用途 |
|---|---------|------|--------|------|------|
| 1 | fine_grained_pruning_analysis | 极高 | ✅ | ~3MB | 细粒度无损剪枝分析 |
| 2 | layer_pruning_analysis | 极高 | ✅ | ~500KB | 逐层剪枝（5-30%） |
| 3 | layer_pruning_analysis_high_sparsity | 极高 | ✅ | ~400KB | 逐层剪枝（40-90%） |
| 4 | gamma_adaptive_pruning_first_order | 高 | ✅ | ~400KB | 自适应剪枝 |
| 5 | pruning_config_test | 中高 | ✅ | ~24KB | 配置验证 |
| 6 | test_bert_sst2 | 中高 | ✅ | ~20KB | BERT-SST2 |
| 7 | test_bert_mnli | 中高 | ✅ | ~16KB | BERT-MNLI |
| 8 | test_bert_stsb | 中高 | ✅ | ~20KB | BERT-STS-B |

**小计**: ~5MB

---

### ⭐⭐ 探索性实验 (12个)

| # | 实验名称 | 价值 | 问题 | 大小 |
|---|---------|------|------|------|
| 9 | blocks_importance | 中 | 缺报告 | 5.5MB |
| 10 | blocks_importance_abs | 中 | 缺报告 | 5.6MB |
| 11 | blocks_importance_abs_fitted | 中 | 缺报告 | 7.9MB |
| 12 | blocks_importance_first_order | 中 | 缺报告 | 5.7MB |
| 13 | blocks_importance_gamma_fitted | 中 | 缺报告 | 8.7MB |
| 14 | blocks_importance_hvp | 中 | 缺报告 | 5.4MB |
| 15 | blocks_importance_hvp_50batches | 中 | 缺报告 | 5.3MB |
| 16 | blocks_importance_hvp_abs | 中 | 缺报告 | 5.8MB |
| 17 | importance_distribution | 中 | - | 872KB |
| 18 | model_structure_visualization | 中 | - | 972KB |
| 19 | outlier_analysis | 中 | - | 1.4MB |
| 20 | checkpoint_visualization | 低 | - | 1.4MB |

**小计**: ~54MB

---

### ⚠️ 重复/过时实验 (11个)

| # | 实验名称 | 问题 | 建议 |
|---|---------|------|------|
| 21 | gamma_adaptive_pruning | 重复 | 删除 |
| 22 | gamma_adaptive_pruning_no_emb | 重复 | 删除 |
| 23 | gamma_adaptive_pruning_multi | 重复 | 删除 |
| 24 | gamma_adaptive_pruning_first_order_multi | 重复 | 删除 |
| 25 | hvp_pruning_multi | 未完成 | 删除 |
| 26 | prune_negative | 未完成 | 删除 |
| 27 | compare_three_methods | 数据不足 | 删除 |
| 28 | compare_first_vs_second_order_direct | 数据不足 | 删除 |
| 29 | pruning_gamma_config_test | 重复 | 删除 |
| 30 | bert_fine_grained_pruning_analysis | 缺报告 | 整合 |
| 31 | test_bert_pruning | 重复 | 删除 |

**小计**: ~5MB

---

### ❌ 空目录 (5个)

| # | 目录名称 | 建议 |
|---|---------|------|
| 32 | checkpoints | 删除 |
| 33 | figures | 删除 |
| 34 | compare_first_vs_second_order_direct/20260121_103104 | 删除 |
| 35 | hvp_pruning_multi/20260121_110757 | 删除 |
| 36 | hvp_pruning_multi/20260121_110604 | 删除 |

---

### 📁 配置库 (1个)

| # | 目录名称 | 内容 | 建议 |
|---|---------|------|------|
| 37 | pruning_configs | 6个JSON配置 | 保留 |

---

## 论文可用数据文件

### 表格数据

1. `layer_pruning_analysis/pruning_results.csv` - 逐层剪枝结果（24行）
2. `layer_pruning_analysis_high_sparsity/pruning_results_high_sparsity.csv` - 高稀疏度结果（18行）
3. `fine_grained_pruning_analysis/lossless_sparsity_summary.csv` - 无损剪枝摘要
4. `fine_grained_pruning_analysis/fine_grained_pruning_results.csv` - 细粒度结果（138KB）
5. `gamma_adaptive_pruning_first_order/layer_prune_ratios.csv` - 自适应剪枝比例
6. `test_bert_sst2/pruning_stats.csv` - BERT-SST2统计
7. `test_bert_mnli/pruning_stats.csv` - BERT-MNLI统计
8. `test_bert_stsb/pruning_stats.csv` - BERT-STS-B统计

### 图表文件

1. `layer_pruning_analysis/loss_vs_sparsity.png` - 损失vs稀疏度
2. `layer_pruning_analysis/loss_increase_vs_sparsity.png` - 损失增量
3. `layer_pruning_analysis_high_sparsity/loss_vs_sparsity_high.png` - 高稀疏度损失
4. `fine_grained_pruning_analysis/lossless_sparsity_heatmap.png` - 无损剪枝热力图
5. `fine_grained_pruning_analysis/block_comparison.png` - 块对比图
6. `gamma_adaptive_pruning_first_order/param_group_prune_ratio_heatmap.png` - 参数组热力图
7. `outlier_analysis/outlier_analysis.png` - 异常值分析

### 报告文件

1. `layer_pruning_analysis/RESULTS_ANALYSIS.md` - 逐层剪枝分析报告
2. `layer_pruning_analysis_high_sparsity/RESULTS_ANALYSIS_HIGH_SPARSITY.md` - 高稀疏度报告
3. `fine_grained_pruning_analysis/ANALYSIS_REPORT.md` - 细粒度分析报告
4. `pruning_config_test/REPORT.md` - 配置测试报告
5. `test_bert_sst2/REPORT.md` - BERT-SST2报告
6. `test_bert_mnli/REPORT.md` - BERT-MNLI报告
7. `test_bert_stsb/REPORT.md` - BERT-STS-B报告

---

## 关键发现摘要

### GPT-2剪枝敏感度

| 层类型 | 最大安全稀疏度 | 临界点 | 冗余度 |
|--------|----------------|--------|--------|
| Embedding | 5% | 5-10% | 极低 |
| Attention | 30-35% | 30-40% | 中等 |
| MLP | 60% | 60-70% | 极高 |
| LayerNorm | 50% | 50-60% | 中高 |

### 细粒度无损剪枝

| 层类型 | 平均无损剪枝 | 最大 | 最小 |
|--------|--------------|------|------|
| Attention Proj | 81.2% | 90% | 70% |
| Attention QKV | 57.9% | 90% | 35% |
| MLP Proj | 52.1% | 80% | 40% |
| MLP FC | 38.8% | 55% | 35% |
| LayerNorm 1 | 30.8% | 90% | 15% |
| LayerNorm 2 | 12.1% | 20% | 5% |

### BERT剪枝结果

| 任务 | 稀疏度 | 基线损失 | 剪枝后损失 | 损失增量 |
|------|--------|----------|------------|----------|
| SST-2 | 25% | 0.745 | 0.834 | +11.9% |
| MNLI | 25% | - | - | - |
| STS-B | 25% | - | - | - |

---

## 存储优化建议

### 当前状态
- **总大小**: 64MB
- **论文核心**: ~5MB (8%)
- **探索性**: ~54MB (84%)
- **重复/过时**: ~5MB (8%)

### 优化后
- **保留**: ~5MB (论文核心)
- **归档**: ~54MB (探索性，可压缩)
- **删除**: ~5MB (重复/过时/空目录)

### 预期效果
- 核心结果目录清晰
- 节省5MB空间
- 探索性实验可选归档

---

## 下一步行动

### 立即执行
1. ✅ 创建 `results/paper_results/` 目录
2. ✅ 移动8个核心实验到paper_results
3. ✅ 删除5个空目录
4. ✅ 删除11个重复/过时实验

### 可选执行
5. 📦 创建 `results/archive/` 并归档探索性实验
6. 📝 为blocks_importance系列创建统一分析报告
7. 📝 补充BERT实验的详细分析

### 后续工作
8. 🔬 完成缺失的压缩方法对比实验
9. 🔬 完成ResNet + CIFAR实验
10. 📊 准备论文图表和表格

---

**清单完成时间**: 2026-02-26  
**详细报告**: 见 `RESULTS_ANALYSIS_REPORT.md`
