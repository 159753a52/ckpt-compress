# 项目清理与重组方案

**生成时间**: 2026-02-26  
**目标**: 为新实验计划（容错训练 + checkpoint compression）清理和重组项目结构

---

## 一、项目现状诊断

### 1.1 根目录污染（9个临时文档）

```
BERT_INTEGRATION_REPORT.txt
BERT_QUICKSTART.md
GPT2_MEDIUM_SUMMARY.md
FILES_CREATED.md
PROJECT_ANALYSIS.md
WORK_SUMMARY.md
DEBUG_MODIFICATIONS.md
QUICK_DEBUG_GUIDE.md
```

**问题**: 这些是开发过程中的临时记录，应该归档或整合到正式文档中。

### 1.2 experiments/scripts/ 目录混乱（35个脚本 + 14个文档）

**脚本分类统计**:
- 训练脚本: 7个（核心功能）
- 压缩脚本: 1个（核心功能）
- Gamma自适应剪枝: 4个（2个核心 + 2个重复）
- 细粒度分析: 2个（核心功能）
- 配置生成/测试: 3个（核心功能）
- 重要性分析: 10个（大部分是探索性实验）
- 对比实验: 8个（大部分功能重复）
- 工具脚本: 4个（部分可归档）

**文档污染**:
```
PRUNING_CONFIGS_COMPARISON.md
PRUNING_CONFIG_TEST_REFERENCE.txt
PRUNING_CONFIG_TEST_SUMMARY.md
PRUNING_TOOLS_SUMMARY.md
QUICK_REFERENCE.txt
README_ALL_BLOCKS_PRUNING.md
README_FINE_GRAINED_PRUNING.md
README_PRUNING_CONFIG_TEST.md
README_layer_pruning_analysis.md
SUMMARY.md
TEST_PRUNING_CONFIG_UPDATE.md
UPDATE_NOTES.md
UPDATE_SUMMARY.txt
VERIFICATION.md
```

### 1.3 results/ 目录混乱（37个子目录）

**保留价值高的实验**:
- `test_bert_sst2/`, `test_bert_mnli/`, `test_bert_stsb/` - BERT实验结果
- `fine_grained_pruning_analysis/` - 细粒度无损剪枝分析
- `layer_pruning_analysis/`, `layer_pruning_analysis_high_sparsity/` - 层级分析

**可归档的探索性实验（30+个）**:
- `blocks_importance*` (8个) - 早期block级别重要性分析
- `gamma_adaptive_pruning*` (4个) - Gamma剪枝的多个版本
- `compare_*` (3个) - 方法对比实验
- `hvp_pruning_multi/` - HVP方法（未采用）
- `prune_negative/` - 负值剪枝探索
- 其他单次分析实验

---

## 二、脚本功能详细分析

### 2.1 训练脚本（保留 - 7个）

#### ✅ train_cv.py
- **功能**: 训练CV模型（ResNet18/50）在CIFAR-10/100上
- **价值**: 核心训练框架，支持检查点保存、早停、学习率调度
- **建议**: **保留** - 对应实验计划中的ViT-L/32实验

#### ✅ train_nlp.py
- **功能**: 训练NLP模型（GPT-2 Small/Medium）在WikiText-2/103上
- **价值**: 核心训练框架
- **建议**: **保留** - 对应实验计划中的GPT-2 Medium实验

#### ✅ finetune_gpt2_1000steps.py
- **功能**: GPT-2 Small在WikiText-103上微调1000步
- **价值**: 生成固定步数的检查点，供剪枝实验使用
- **建议**: **保留并重命名** → `finetune_gpt2_medium.py`（扩展支持Medium）

#### ✅ finetune_bert_large_sst2.py
- **功能**: BERT-Large在SST-2上微调（情感分类）
- **价值**: BERT实验基础，已有实验结果
- **建议**: **保留** - 对应实验计划中的BERT-Large实验

#### ✅ finetune_bert_large_mnli.py
- **功能**: BERT-Large在MNLI上微调（自然语言推理）
- **价值**: BERT实验基础，已有实验结果
- **建议**: **保留** - 对应实验计划中的BERT-Large实验

#### ✅ finetune_bert_large_stsb.py
- **功能**: BERT-Large在STS-B上微调（语义相似度回归）
- **价值**: BERT实验基础，已有实验结果
- **建议**: **保留** - 对应实验计划中的BERT-Large实验

---

### 2.2 压缩脚本（保留 - 1个）

#### ✅ compress_and_resume.py
- **功能**: 使用不同方法（ExCP/Inshrinkerator/PredictiveResidual）压缩和解压检查点
- **价值**: 项目核心功能，演示检查点压缩和恢复
- **建议**: **保留并扩展** - 需要添加容错训练模拟功能

---

### 2.3 Gamma自适应剪枝（保留2个，归档2个）

#### ✅ gamma_adaptive_pruning.py
- **功能**: 基于Gamma分布的自适应剪枝（一阶+二阶重要性）
- **重要性公式**: `d_i = |g_i * θ_i| + α * |v_i * θ_i²|`
- **价值**: **AdamPrune核心方法**，使用Gamma分布拟合 + 全局阈值求解
- **建议**: **保留** - 这是你的核心贡献

#### ✅ gamma_adaptive_pruning_first_order.py
- **功能**: 基于Gamma分布的自适应剪枝（纯一阶重要性）
- **重要性公式**: `d_i = |g_i * θ_i|`
- **价值**: 消融实验基础，验证二阶项的作用
- **建议**: **保留** - 对应实验计划Table 3的消融实验

#### 📦 gamma_adaptive_pruning_multi_ratio.py
- **功能**: 测试多个全局剪枝率（一阶+二阶）
- **问题**: 功能重复，可通过循环调用主脚本实现
- **建议**: **归档** - 可用工作流脚本替代

#### 📦 gamma_adaptive_pruning_first_order_multi_ratio.py
- **功能**: 测试多个全局剪枝率（纯一阶）
- **问题**: 功能重复
- **建议**: **归档** - 可用工作流脚本替代

---

### 2.4 细粒度分析（保留 - 2个）

#### ✅ fine_grained_pruning_analysis.py
- **功能**: GPT-2细粒度无损剪枝分析（attn_qkv, attn_proj, mlp_fc, mlp_proj, ln_1, ln_2）
- **价值**: 
  - 分析每个参数组的无损剪枝比例
  - 生成热力图和参数数量统计
  - 为配置生成提供数据基础
- **已有结果**: `results/fine_grained_pruning_analysis/ANALYSIS_REPORT.md`
- **建议**: **保留** - 对应实验计划Fig 5（层级剪枝率分布）

#### ✅ bert_fine_grained_pruning_analysis.py
- **功能**: BERT-Large细粒度无损剪枝分析（attn_q/k/v/out, ffn_fc/proj, ln_attn/ffn）
- **价值**: BERT模型的细粒度剪枝分析，支持分类和回归任务
- **建议**: **保留** - BERT实验的分析工具

---

### 2.5 配置生成与测试（保留 - 3个）

#### ✅ generate_pruning_config.py
- **功能**: 从无损剪枝分析结果生成目标剪枝率配置
- **价值**: 根据无损剪枝分析自动生成剪枝配置，支持目标剪枝率分配
- **建议**: **保留** - 配置生成工具

#### ✅ generate_global_pruning_config.py
- **功能**: 生成全局统一剪枝配置（所有参数组使用相同剪枝率）
- **价值**: 生成Baseline配置（Magnitude + Uniform）
- **建议**: **保留** - 对应实验计划Table 1的Baseline

#### ✅ test_pruning_config.py
- **功能**: 测试自定义剪枝配置（支持GPT-2和BERT，支持语言模型/分类/回归任务）
- **价值**: 灵活的剪枝配置测试框架，验证剪枝配置的有效性
- **建议**: **保留** - 核心测试工具

---

### 2.6 方法对比实验（保留1个，归档7个）

#### ✅ compare_first_vs_second_order.py
- **功能**: 一阶 vs 一阶+二阶重要性方法对比（使用相同数据批次）
- **价值**: 公平对比两种重要性计算方法，验证二阶项的贡献
- **建议**: **保留** - 对应实验计划Table 3的消融实验

#### 📦 compare_first_vs_second_order_checkpoint.py
- **功能**: 从检查点加载后对比一阶 vs 二阶
- **问题**: 功能与主脚本重复
- **建议**: **归档**

#### 📦 compare_first_vs_second_order_direct.py
- **功能**: 直接对比（不使用检查点）
- **问题**: 功能与主脚本重复
- **建议**: **归档**

#### 📦 compare_three_methods.py
- **功能**: 对比三种方法（一阶、一阶+二阶、HVP）
- **价值**: 已验证Adam近似方法的有效性
- **建议**: **归档** - 对比已完成

#### 📦 compare_gamma_exponential_fit.py
- **功能**: 对比Gamma分布 vs 指数分布拟合
- **价值**: 已确定Gamma分布更适合
- **建议**: **归档** - 分布选择已完成

#### 📦 compare_hvp_batches.py
- **功能**: 对比不同批次数量对HVP计算的影响
- **价值**: HVP方法计算成本高，未作为主要方法
- **建议**: **归档**

#### 📦 hvp_pruning_multi_ratio.py
- **功能**: 使用HVP方法测试多个剪枝率
- **价值**: HVP方法未采用
- **建议**: **归档**

---

### 2.7 重要性分析（归档 - 10个）

所有这些脚本都是早期探索性实验，用于理解重要性分布特征。


#### 📦 analyze_importance_distribution.py
- **功能**: 计算并可视化GPT-2不同参数组的重要性分布
- **建议**: **归档** - 探索完成

#### 📦 analyze_all_blocks_importance.py
- **功能**: 为每个transformer block生成独立的重要性分布图（保留符号）
- **建议**: **归档** - 可视化探索

#### 📦 analyze_all_blocks_importance_abs.py
- **功能**: 同上，但使用绝对值
- **建议**: **归档** - 功能重复

#### 📦 analyze_all_blocks_importance_first_order.py
- **功能**: 纯一阶重要性的block分析
- **建议**: **归档** - 功能重复

#### 📦 analyze_all_blocks_importance_hvp.py
- **功能**: 使用HVP（Hessian-Vector Product）计算精确二阶重要性
- **建议**: **归档** - HVP计算成本高，未采用

#### 📦 analyze_all_blocks_importance_hvp_abs.py
- **功能**: HVP方法的绝对值版本
- **建议**: **归档** - 功能重复

#### 📦 analyze_single_tensor_importance.py
- **功能**: 分析单个张量的重要性分布
- **建议**: **归档** - 调试工具

#### 📦 analyze_numerical_sensitivity.py
- **功能**: 分析数值敏感性（浮点精度、量化等）
- **建议**: **归档** - 一次性分析

#### 📦 analyze_numerical_sensitivity_v2.py
- **功能**: 数值敏感性分析v2
- **建议**: **归档** - 版本迭代

#### 📦 layer_pruning_analysis.py
- **功能**: 逐层剪枝对损失的影响分析
- **建议**: **归档** - 被fine_grained_pruning_analysis.py取代

#### 📦 layer_pruning_analysis_high_sparsity.py
- **功能**: 高稀疏度（50%-90%）的逐层剪枝分析
- **建议**: **归档** - 一次性探索

---

### 2.8 工具脚本（保留2个，归档2个）

#### ✅ visualize_model_structure.py
- **功能**: 可视化GPT-2 Small和BERT-Large的参数量分布
- **价值**: 帮助理解模型结构，用于论文图表
- **建议**: **保留** - 有用的分析工具

#### ✅ load_finetuned_checkpoint.py
- **功能**: 演示如何加载和使用微调后的检查点
- **价值**: 提供检查点加载的示例代码
- **建议**: **保留** - 文档示例

#### 📦 generate_csv_data.py
- **功能**: 生成重要性得分的CSV数据用于诊断
- **建议**: **归档** - 调试完成

#### 📦 generate_block_importance_tsv.py
- **功能**: 生成block重要性的TSV文件
- **建议**: **归档** - 一次性导出

---

## 三、脚本清理统计

### 3.1 保留脚本（18个）

**训练类（7个）**:
1. train_cv.py
2. train_nlp.py
3. finetune_gpt2_1000steps.py → 重命名为 finetune_gpt2_medium.py
4. finetune_bert_large_sst2.py
5. finetune_bert_large_mnli.py
6. finetune_bert_large_stsb.py

**压缩类（1个）**:
7. compress_and_resume.py

**核心方法（2个）**:
8. gamma_adaptive_pruning.py
9. gamma_adaptive_pruning_first_order.py

**分析工具（2个）**:
10. fine_grained_pruning_analysis.py
11. bert_fine_grained_pruning_analysis.py

**配置工具（3个）**:
12. generate_pruning_config.py
13. generate_global_pruning_config.py
14. test_pruning_config.py

**对比实验（1个）**:
15. compare_first_vs_second_order.py

**工具脚本（2个）**:
16. visualize_model_structure.py
17. load_finetuned_checkpoint.py

### 3.2 归档脚本（17个）

- gamma_adaptive_pruning_multi_ratio.py
- gamma_adaptive_pruning_first_order_multi_ratio.py
- layer_pruning_analysis.py
- layer_pruning_analysis_high_sparsity.py
- analyze_importance_distribution.py
- analyze_all_blocks_importance.py (及其4个变体)
- analyze_single_tensor_importance.py
- compare_first_vs_second_order_checkpoint.py
- compare_first_vs_second_order_direct.py
- compare_three_methods.py
- compare_gamma_exponential_fit.py
- compare_hvp_batches.py
- hvp_pruning_multi_ratio.py
- analyze_numerical_sensitivity.py
- analyze_numerical_sensitivity_v2.py
- generate_csv_data.py
- generate_block_importance_tsv.py

---

## 四、results/ 目录清理

### 4.1 保留实验结果（6个目录）

#### ✅ test_bert_sst2/
- **内容**: BERT-Large在SST-2上的剪枝测试
- **价值**: 已有完整实验报告
- **建议**: **保留** - 对应BERT实验

#### ✅ test_bert_mnli/
- **内容**: BERT-Large在MNLI上的剪枝测试
- **价值**: 已有完整实验报告
- **建议**: **保留** - 对应BERT实验

#### ✅ test_bert_stsb/
- **内容**: BERT-Large在STS-B上的剪枝测试
- **价值**: 已有完整实验报告
- **建议**: **保留** - 对应BERT实验

#### ✅ fine_grained_pruning_analysis/
- **内容**: GPT-2细粒度无损剪枝分析结果
- **价值**: 
  - 包含完整的ANALYSIS_REPORT.md
  - 各参数组的无损剪枝比例数据
  - 热力图和统计信息
- **建议**: **保留** - 重要分析结果

#### ✅ layer_pruning_analysis/
- **内容**: 逐层剪枝分析结果
- **建议**: **保留** - 有参考价值

#### ✅ layer_pruning_analysis_high_sparsity/
- **内容**: 高稀疏度剪枝分析结果
- **建议**: **保留** - 有参考价值

### 4.2 归档实验结果（31个目录）

**Block重要性分析（8个）**:
- blocks_importance/
- blocks_importance_abs/
- blocks_importance_abs_fitted/
- blocks_importance_first_order/
- blocks_importance_gamma_fitted/
- blocks_importance_hvp/
- blocks_importance_hvp_50batches/
- blocks_importance_hvp_abs/

**Gamma自适应剪枝（4个）**:
- gamma_adaptive_pruning/
- gamma_adaptive_pruning_first_order/
- gamma_adaptive_pruning_first_order_multi/
- gamma_adaptive_pruning_multi/
- gamma_adaptive_pruning_no_emb/

**方法对比（3个）**:
- compare_first_vs_second_order_direct/
- compare_three_methods/
- hvp_pruning_multi/

**其他探索性实验（16个）**:
- importance_distribution/
- single_tensor_importance/
- prune_negative/
- checkpoint_visualization/
- outlier_analysis/
- model_structure_visualization/
- pruning_config_test/
- pruning_gamma_config_test/
- pruning_configs/
- checkpoints/
- figures/
- bert_fine_grained_pruning_analysis/

**临时文件**:
- *.log, *.tsv, *.json, *.md (根目录下的临时文件)

---

## 五、执行方案

### 5.1 备份（必须先执行）

```bash
cd /lihongliang/fangzl
tar -czf ckpt-compress-backup-$(date +%Y%m%d-%H%M%S).tar.gz ckpt-compress/
```

### 5.2 创建归档目录

```bash
cd /lihongliang/fangzl/ckpt-compress
mkdir -p archive/{docs,scripts,results}
```

### 5.3 归档根目录文档

```bash
mv BERT_INTEGRATION_REPORT.txt BERT_QUICKSTART.md \
   GPT2_MEDIUM_SUMMARY.md FILES_CREATED.md \
   PROJECT_ANALYSIS.md WORK_SUMMARY.md \
   DEBUG_MODIFICATIONS.md QUICK_DEBUG_GUIDE.md \
   archive/docs/
```

### 5.4 归档 experiments/scripts/ 中的文档

```bash
mv experiments/scripts/PRUNING_*.md \
   experiments/scripts/README_*.md \
   experiments/scripts/SUMMARY.md \
   experiments/scripts/QUICK_REFERENCE.txt \
   experiments/scripts/UPDATE_*.md \
   experiments/scripts/UPDATE_*.txt \
   experiments/scripts/VERIFICATION.md \
   experiments/scripts/TEST_*.md \
   archive/scripts/
```

### 5.5 归档 experiments/scripts/ 中的脚本

```bash
# 归档重要性分析脚本
mv experiments/scripts/analyze_*.py archive/scripts/

# 归档对比实验脚本（保留compare_first_vs_second_order.py）
mv experiments/scripts/compare_first_vs_second_order_checkpoint.py \
   experiments/scripts/compare_first_vs_second_order_direct.py \
   experiments/scripts/compare_three_methods.py \
   experiments/scripts/compare_gamma_exponential_fit.py \
   experiments/scripts/compare_hvp_batches.py \
   experiments/scripts/hvp_pruning_multi_ratio.py \
   archive/scripts/

# 归档多比例测试脚本
mv experiments/scripts/gamma_adaptive_pruning_multi_ratio.py \
   experiments/scripts/gamma_adaptive_pruning_first_order_multi_ratio.py \
   archive/scripts/

# 归档工具脚本
mv experiments/scripts/generate_csv_data.py \
   experiments/scripts/generate_block_importance_tsv.py \
   archive/scripts/

# 归档shell脚本
mv experiments/scripts/*.sh archive/scripts/
```

### 5.6 归档 results/ 中的实验结果

```bash
# 归档block重要性分析
mv results/blocks_importance* archive/results/

# 归档gamma自适应剪枝实验
mv results/gamma_adaptive_pruning/ \
   results/gamma_adaptive_pruning_first_order/ \
   results/gamma_adaptive_pruning_first_order_multi/ \
   results/gamma_adaptive_pruning_multi/ \
   results/gamma_adaptive_pruning_no_emb/ \
   archive/results/

# 归档方法对比实验
mv results/compare_* results/hvp_pruning_multi/ archive/results/

# 归档其他探索性实验
mv results/importance_distribution/ \
   results/single_tensor_importance/ \
   results/prune_negative/ \
   results/checkpoint_visualization/ \
   results/outlier_analysis/ \
   results/model_structure_visualization/ \
   results/pruning_config_test/ \
   results/pruning_gamma_config_test/ \
   results/pruning_configs/ \
   results/checkpoints/ \
   results/figures/ \
   results/bert_fine_grained_pruning_analysis/ \
   archive/results/

# 归档临时文件
mv results/*.log results/*.tsv results/*.json results/*.md archive/results/
```


---

## 六、重组实验目录结构

### 6.1 新的目录结构

```
experiments/
├── README.md                    # 实验说明
├── configs/                     # 配置文件（按模型组织）
│   ├── gpt2_medium/
│   │   ├── finetune.yaml
│   │   └── pruning_*.json
│   ├── pythia_410m/
│   ├── bert_large/
│   │   ├── sst2.yaml
│   │   ├── mnli.yaml
│   │   └── stsb.yaml
│   └── vit_l32/
├── scripts/
│   ├── finetune/                # 微调脚本
│   │   ├── finetune_gpt2_medium.py
│   │   ├── finetune_pythia_410m.py
│   │   ├── finetune_bert_large.py
│   │   └── finetune_vit_l32.py
│   ├── compress/                # 压缩脚本
│   │   ├── compress_checkpoint.py
│   │   └── decompress_checkpoint.py
│   ├── prune/                   # 剪枝脚本
│   │   ├── gamma_adaptive_pruning.py
│   │   ├── gamma_adaptive_pruning_first_order.py
│   │   └── magnitude_pruning.py
│   ├── evaluate/                # 评估脚本
│   │   ├── evaluate_quality.py
│   │   └── evaluate_compression_ratio.py
│   ├── analysis/                # 分析脚本
│   │   ├── fine_grained_pruning_analysis.py
│   │   ├── bert_fine_grained_pruning_analysis.py
│   │   ├── analyze_gamma_fitting.py
│   │   ├── analyze_layer_rates.py
│   │   └── visualize_model_structure.py
│   ├── config/                  # 配置生成脚本
│   │   ├── generate_pruning_config.py
│   │   ├── generate_global_pruning_config.py
│   │   └── test_pruning_config.py
│   └── comparison/              # 对比实验
│       └── compare_first_vs_second_order.py
└── workflows/                   # 端到端工作流
    ├── run_fault_tolerant_training.py
    ├── run_ablation_study.py
    └── run_pareto_analysis.py

results/
├── gpt2_medium/
│   ├── baseline_magnitude/
│   ├── baseline_inshrinkerator/
│   ├── baseline_excp/
│   └── ours/
├── pythia_410m/
├── bert_large/
│   ├── sst2/
│   ├── mnli/
│   └── stsb/
├── vit_l32/
└── ablation/
    ├── importance_score/
    └── rate_allocation/
```

### 6.2 重组命令

```bash
# 创建新目录结构
mkdir -p experiments/configs/{gpt2_medium,pythia_410m,bert_large,vit_l32}
mkdir -p experiments/scripts/{finetune,compress,prune,evaluate,analysis,config,comparison}
mkdir -p experiments/workflows
mkdir -p results/{gpt2_medium,pythia_410m,bert_large,vit_l32,ablation}/{baseline_magnitude,baseline_inshrinkerator,baseline_excp,ours}
mkdir -p results/bert_large/{sst2,mnli,stsb}
mkdir -p results/ablation/{importance_score,rate_allocation}

# 移动训练脚本
mv experiments/scripts/train_cv.py experiments/scripts/finetune/
mv experiments/scripts/train_nlp.py experiments/scripts/finetune/
mv experiments/scripts/finetune_gpt2_1000steps.py experiments/scripts/finetune/finetune_gpt2_medium.py
mv experiments/scripts/finetune_bert_large_*.py experiments/scripts/finetune/

# 移动压缩脚本
mv experiments/scripts/compress_and_resume.py experiments/scripts/compress/

# 移动剪枝脚本
mv experiments/scripts/gamma_adaptive_pruning.py experiments/scripts/prune/
mv experiments/scripts/gamma_adaptive_pruning_first_order.py experiments/scripts/prune/

# 移动分析脚本
mv experiments/scripts/fine_grained_pruning_analysis.py experiments/scripts/analysis/
mv experiments/scripts/bert_fine_grained_pruning_analysis.py experiments/scripts/analysis/
mv experiments/scripts/visualize_model_structure.py experiments/scripts/analysis/

# 移动配置脚本
mv experiments/scripts/generate_pruning_config.py experiments/scripts/config/
mv experiments/scripts/generate_global_pruning_config.py experiments/scripts/config/
mv experiments/scripts/test_pruning_config.py experiments/scripts/config/

# 移动对比实验
mv experiments/scripts/compare_first_vs_second_order.py experiments/scripts/comparison/

# 移动示例脚本到examples/
mv experiments/scripts/load_finetuned_checkpoint.py examples/

# 移动现有实验结果
mv results/test_bert_sst2/* results/bert_large/sst2/
mv results/test_bert_mnli/* results/bert_large/mnli/
mv results/test_bert_stsb/* results/bert_large/stsb/
rmdir results/test_bert_*

mv results/fine_grained_pruning_analysis results/ablation/
mv results/layer_pruning_analysis* results/ablation/
```

---

## 七、需要创建的新脚本

根据实验计划，需要创建以下新脚本：

### 7.1 容错训练主脚本

**文件**: `experiments/workflows/run_fault_tolerant_training.py`

**功能**:
- 模拟多次checkpoint压缩和恢复（5-10次）
- 支持所有baseline方法对比（Magnitude, Inshrinkerator, ExCP, Ours）
- 自动记录loss曲线和质量指标
- 生成Table 1和Fig 6

**关键参数**:
```python
--model: gpt2-medium / pythia-410m / bert-large / vit-l32
--dataset: wikitext103 / alpaca / glue / imagenet
--method: magnitude / inshrinkerator / excp / ours
--prune_ratio: 0.5 / 0.7 / 0.8 / 0.9
--num_recoveries: 5-10
--checkpoint_freq: N步保存一次
```

### 7.2 消融实验脚本

**文件**: `experiments/workflows/run_ablation_study.py`

**功能**:
- 拆解importance score和rate allocation的独立贡献
- 生成Table 3

**实验组合**:
1. Magnitude + Uniform
2. 1st-order + Uniform
3. 1st+2nd order + Uniform
4. 1st-order + Dist-aware
5. 1st+2nd order + Dist-aware (Full)

### 7.3 Pareto曲线生成

**文件**: `experiments/workflows/run_pareto_analysis.py`

**功能**:
- 扫描不同剪枝比例（0%-95%，步长5%）
- 生成Fig 4（Pareto曲线）

### 7.4 Gamma拟合验证

**文件**: `experiments/scripts/analysis/analyze_gamma_fitting.py`

**功能**:
- 选3个代表性层（Attention Q/K/V, MLP, LayerNorm）
- 画damage score直方图 + 拟合的Gamma PDF曲线
- 生成Fig 3

### 7.5 层级剪枝率分布可视化

**文件**: `experiments/scripts/analysis/analyze_layer_rates.py`

**功能**:
- 柱状图：X轴为层编号，Y轴为该层实际剪枝率
- 对比Uniform vs Ours
- 生成Fig 5

---

## 八、文档整理

### 8.1 根目录保留

- `CLAUDE.md` - 项目指南（保留）
- `pyproject.toml` - 项目配置（保留）
- `README.md` - 如果没有，需要创建

### 8.2 docs/ 目录保留

- `ARCHITECTURE.md`
- `EXPERIMENTS.md`
- `DATA_PREPARATION.md`
- `BERT_USAGE.md`

### 8.3 需要创建的新文档

**experiments/README.md**:
```markdown
# 实验说明

## 目录结构
- `configs/`: 各模型的配置文件
- `scripts/`: 实验脚本（按功能分类）
- `workflows/`: 端到端实验工作流

## 快速开始
见 `/plan/experiment_plan.md`

## 核心实验

### P0: GPT-2 Medium容错训练
```bash
python experiments/workflows/run_fault_tolerant_training.py \
  --model gpt2-medium \
  --dataset wikitext103 \
  --method ours \
  --prune_ratio 0.7 \
  --num_recoveries 10
```

### P1: Pythia-410M指令微调
```bash
python experiments/workflows/run_fault_tolerant_training.py \
  --model pythia-410m \
  --dataset alpaca \
  --method ours \
  --prune_ratio 0.7 \
  --num_recoveries 10
```

### 消融实验
```bash
python experiments/workflows/run_ablation_study.py \
  --model gpt2-medium \
  --dataset wikitext103
```

### Pareto曲线
```bash
python experiments/workflows/run_pareto_analysis.py \
  --model gpt2-medium \
  --dataset wikitext103
```
```

---

## 九、清理后的目录结构预览

```
ckpt-compress/
├── CLAUDE.md                    # 项目指南
├── README.md                    # 项目说明
├── pyproject.toml
├── plan/
│   ├── experiment_plan.md       # 实验计划
│   └── cleanup_plan.md          # 本文档
├── docs/                        # 核心文档（4个）
│   ├── ARCHITECTURE.md
│   ├── EXPERIMENTS.md
│   ├── DATA_PREPARATION.md
│   └── BERT_USAGE.md
├── src/ckpt_compress/           # 源代码（不变）
├── tests/                       # 测试（不变）
├── data/                        # 数据集（不变）
├── checkpoints/                 # 训练好的检查点（不变）
├── examples/                    # 示例脚本
│   └── load_finetuned_checkpoint.py
├── experiments/
│   ├── README.md
│   ├── configs/                 # 按模型组织
│   ├── scripts/                 # 按功能分类（7个子目录）
│   └── workflows/               # 端到端工作流（3个新脚本）
├── results/                     # 按模型+方法组织（清爽）
│   ├── gpt2_medium/
│   ├── pythia_410m/
│   ├── bert_large/
│   ├── vit_l32/
│   └── ablation/
└── archive/                     # 历史文档和实验
    ├── docs/                    # 9个临时文档
    ├── scripts/                 # 17个旧脚本 + 14个文档
    └── results/                 # 31个旧实验

```

---

## 十、执行清单

### ✅ 阶段1：备份（必须）
- [ ] 创建完整备份：`tar -czf ckpt-compress-backup-$(date +%Y%m%d-%H%M%S).tar.gz ckpt-compress/`

### ✅ 阶段2：归档（安全操作）
- [ ] 创建归档目录：`mkdir -p archive/{docs,scripts,results}`
- [ ] 归档根目录文档（9个）
- [ ] 归档experiments/scripts/文档（14个）
- [ ] 归档experiments/scripts/脚本（17个）
- [ ] 归档results/实验结果（31个目录）

### ✅ 阶段3：重组（结构优化）
- [ ] 创建新的experiments/目录结构
- [ ] 创建新的results/目录结构
- [ ] 移动保留的脚本到新位置（18个）
- [ ] 移动保留的实验结果到新位置（6个目录）

### ✅ 阶段4：创建新脚本（对应实验计划）
- [ ] `experiments/workflows/run_fault_tolerant_training.py`
- [ ] `experiments/workflows/run_ablation_study.py`
- [ ] `experiments/workflows/run_pareto_analysis.py`
- [ ] `experiments/scripts/analysis/analyze_gamma_fitting.py`
- [ ] `experiments/scripts/analysis/analyze_layer_rates.py`

### ✅ 阶段5：文档更新
- [ ] 创建`experiments/README.md`
- [ ] 更新`CLAUDE.md`（反映新结构）
- [ ] 创建根目录`README.md`（如果没有）

---

## 十一、关键改进总结

### 11.1 清理效果

**根目录**:
- 清理前：9个临时文档
- 清理后：只保留CLAUDE.md和pyproject.toml

**experiments/scripts/**:
- 清理前：35个脚本 + 14个文档
- 清理后：18个脚本，按功能分类到7个子目录

**results/**:
- 清理前：37个子目录
- 清理后：6个保留目录，按模型+方法组织

### 11.2 结构优势

1. **按模型组织**: configs/和results/按模型分类，对应实验计划
2. **按功能分类**: scripts/按功能分为7类，职责清晰
3. **工作流驱动**: workflows/提供端到端实验脚本
4. **历史可追溯**: archive/保留所有旧内容
5. **便于扩展**: 新增模型只需添加目录

### 11.3 对应实验计划

| 实验计划 | 对应脚本/目录 |
|---------|--------------|
| P0: GPT-2 Medium | `experiments/scripts/finetune/finetune_gpt2_medium.py` |
| P1: Pythia-410M | `experiments/scripts/finetune/finetune_pythia_410m.py` (待创建) |
| P2: BERT-Large | `experiments/scripts/finetune/finetune_bert_large.py` |
| P3: ViT-L/32 | `experiments/scripts/finetune/finetune_vit_l32.py` (待创建) |
| Table 1 | `experiments/workflows/run_fault_tolerant_training.py` (待创建) |
| Table 3 | `experiments/workflows/run_ablation_study.py` (待创建) |
| Fig 3 | `experiments/scripts/analysis/analyze_gamma_fitting.py` (待创建) |
| Fig 4 | `experiments/workflows/run_pareto_analysis.py` (待创建) |
| Fig 5 | `experiments/scripts/analysis/analyze_layer_rates.py` (待创建) |
| Fig 6 | `experiments/workflows/run_fault_tolerant_training.py` (待创建) |

---

## 十二、注意事项

### 12.1 执行顺序

1. **必须先备份**：避免误操作导致数据丢失
2. **先归档后重组**：确保所有文件都有去处
3. **逐步验证**：每个阶段完成后检查文件是否正确移动

### 12.2 安全检查

```bash
# 检查归档是否完整
ls archive/docs/ | wc -l    # 应该是9
ls archive/scripts/ | wc -l  # 应该是31（17个脚本 + 14个文档）
ls archive/results/ | wc -l  # 应该是31+

# 检查保留脚本数量
find experiments/scripts/ -name "*.py" | wc -l  # 应该是18

# 检查保留实验结果
ls results/ | wc -l  # 应该是6（包括ablation）
```

### 12.3 回滚方案

如果清理后发现问题，可以从备份恢复：

```bash
cd /lihongliang/fangzl
rm -rf ckpt-compress/
tar -xzf ckpt-compress-backup-<timestamp>.tar.gz
```

---

## 十三、后续工作

清理完成后，按以下优先级开展实验：

1. **创建工作流脚本**（P0）
   - `run_fault_tolerant_training.py`
   - `run_ablation_study.py`
   - `run_pareto_analysis.py`

2. **运行GPT-2 Medium实验**（P0）
   - 生成Table 1第一行
   - 生成Fig 6的loss曲线

3. **创建分析脚本**（P1）
   - `analyze_gamma_fitting.py` → Fig 3
   - `analyze_layer_rates.py` → Fig 5

4. **扩展到其他模型**（P1-P3）
   - Pythia-410M
   - BERT-Large
   - ViT-L/32

---

**文档结束**
