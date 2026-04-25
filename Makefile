# ckpt-compress Makefile
# 用于简化实验运行和开发流程

.PHONY: help install dev test lint clean all table1 table2 table3 fig4 fig5 fig6 gamma sensitivity violin

# 默认目标
help:
	@echo "ckpt-compress Makefile"
	@echo ""
	@echo "可用目标:"
	@echo "  install     - 安装项目（生产环境）"
	@echo "  dev         - 安装项目（开发环境）"
	@echo "  test        - 运行所有测试"
	@echo "  lint        - 代码检查（black + isort + mypy）"
	@echo "  format      - 格式化代码"
	@echo "  clean       - 清理实验结果和缓存"
	@echo ""
	@echo "实验目标:"
	@echo "  table1      - Table 1: 主实验（importance × allocation 对比）"
	@echo "  table2      - Table 2: 联合压缩（Pruning + INT4 Quantization）"
	@echo "  table3      - Table 3: 消融实验"
	@echo "  fig4        - Figure 4: Pareto 曲线"
	@echo "  fig5        - Figure 5: Importance score 热力图"
	@echo "  fig6        - Figure 6: Fault-tolerant 训练"
	@echo "  gamma       - §9.1-9.2: Gamma 分布验证"
	@echo "  sensitivity - §9.3: 敏感性分析 + 计时分解"
	@echo "  violin      - §9.4: 层分布 Violin 图"
	@echo "  all         - 运行全部实验"

# 安装
install:
	pip install -e .

dev: install
	pip install -e ".[dev]"

# 测试
test:
	pytest tests/

test-cov:
	pytest --cov=dacp --cov-report=html --cov-report=term

test-unit:
	pytest tests/unit/

test-integration:
	pytest tests/integration/

# 代码质量
lint:
	black --check dacp/ baselines/ experiments/
	isort --check-only dacp/ baselines/ experiments/
	mypy dacp/

format:
	black dacp/ baselines/ experiments/
	isort dacp/ baselines/ experiments/

# 实验
table1:
	python experiments/scripts/run_method_comparison.py \
		--model gpt2-medium --dataset wikitext103 --device cuda

table2:
	python experiments/scripts/run_joint_compression.py \
		--model gpt2-medium --dataset wikitext103 --device cuda

table3:
	python experiments/scripts/run_ablation_study.py \
		--model gpt2-medium --dataset wikitext103 --device cuda

fig4:
	python experiments/scripts/run_pareto_curves.py \
		--model gpt2-medium --dataset wikitext103 --device cuda

fig5:
	python experiments/scripts/run_pruning_heatmap.py \
		--model gpt2-medium --dataset wikitext103 --device cuda

fig6:
	python experiments/scripts/run_fault_tolerant_training.py \
		--model gpt2-medium --dataset wikitext103 --device cuda

gamma:
	python experiments/scripts/run_gamma_validation.py \
		--model gpt2-medium --dataset wikitext103 --device cuda

sensitivity:
	python experiments/scripts/run_sensitivity_analysis.py \
		--model gpt2-medium --dataset wikitext103 --device cuda

violin:
	python experiments/scripts/run_layer_distribution_violin.py \
		--model gpt2-medium --dataset wikitext103 --device cuda

all: table1 table2 table3 fig4 fig5 fig6 gamma sensitivity violin
	@echo "所有实验已完成"

# 清理
clean:
	@echo "清理实验结果..."
	rm -rf experiments/results/*
	@echo "清理 Python 缓存..."
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	@echo "清理完成"

# 下载数据和模型
download-data:
	python scripts/download_data.py --all

download-models:
	python scripts/download_models.py --all

download: download-data download-models

# 文档
docs:
	@echo "生成文档（待实现）"
	# sphinx-build -b html docs/ docs/_build/

# 发布
release:
	@echo "发布到 PyPI（待实现）"
	# python -m build
	# twine upload dist/*
