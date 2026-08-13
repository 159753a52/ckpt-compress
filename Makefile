# ckpt-compress Makefile
# 用于简化实验运行和开发流程

.PHONY: help install dev test test-cov test-paper-cov typecheck-paper lint format clean paper-plan wheel

PAPER_PYTHON_FILES := \
	dacp/utils/paths.py \
	experiments/lib/paper_baselines.py \
	experiments/lib/paper_manifest.py \
	experiments/lib/paper_result_validation.py \
	experiments/lib/paper_results.py \
	experiments/lib/paper_runner.py \
	experiments/lib/evaluation.py \
	experiments/lib/losses.py \
	experiments/lib/residual_masks.py \
	experiments/lib/residual_weibull.py \
	experiments/scripts/finetune/finetune_pythia_410m.py \
	experiments/scripts/run_paper_experiments.py \
	scripts/validate_distribution.py \
	tests/test_data.py \
	tests/test_evaluation.py \
	tests/test_experiments_lib_init.py \
	tests/test_finetune_cli.py \
	tests/test_finetune_evaluation.py \
	tests/test_losses.py \
	tests/test_paper_experiments.py \
	tests/test_paper_results.py \
	tests/test_paths.py \
	tests/test_residual_masks.py \
	tests/test_validate_distribution.py
CHANGED_PYTHON_FILES := $(shell git diff HEAD --name-only --diff-filter=ACMR -- '*.py') \
	$(shell git ls-files --others --exclude-standard -- '*.py')
CHECK_PYTHON_FILES := $(sort $(PAPER_PYTHON_FILES) $(CHANGED_PYTHON_FILES))

# 默认目标
help:
	@echo "ckpt-compress Makefile"
	@echo ""
	@echo "可用目标:"
	@echo "  install     - 安装项目（生产环境）"
	@echo "  dev         - 安装项目（开发环境）"
	@echo "  test        - 运行所有测试"
	@echo "  test-cov    - 报告整个 dacp 包覆盖率（当前不设虚假全局门禁）"
	@echo "  test-paper-cov - 对论文核心选择模块执行 90% 覆盖率门禁"
	@echo "  typecheck-paper - 检查全部可发布生产 Python（排除测试与 ExCP upstream）"
	@echo "  lint        - 当前论文路径的编译、diff 与 black 检查"
	@echo "  format      - 格式化代码"
	@echo "  clean       - 仅清理 Python/测试缓存（不删除实验结果）"
	@echo "  paper-plan  - 显示默认论文实验清单，不启动 GPU 任务"
	@echo "  wheel       - 构建可安装 wheel"

# 安装
install:
	pip install -e .

dev: install
	pip install -e ".[dev]"

# 测试
test:
	python -m pytest -q

test-cov:
	python -m pytest --cov=dacp --cov-report=html --cov-report=term --cov-fail-under=0

test-paper-cov:
	python -m pytest -q \
		tests/test_paper_experiments.py tests/test_residual_masks.py tests/test_paths.py \
		--cov=experiments.lib.paper_runner \
		--cov=experiments.lib.residual_masks \
		--cov=dacp.utils.paths \
		--cov-report=term --cov-fail-under=90

typecheck-paper:
	python scripts/list_production_python.py mypy -- \
		--explicit-package-bases --follow-imports=skip \
		--disable-error-code=import-untyped

# 代码质量
lint:
	python -m compileall -q dacp baselines experiments scripts tests
	git diff --check
	python -m isort --check-only $(CHECK_PYTHON_FILES)
	python -m black --check $(CHECK_PYTHON_FILES)
	$(MAKE) typecheck-paper

format:
	python -m isort $(CHECK_PYTHON_FILES)
	python -m black $(CHECK_PYTHON_FILES)

paper-plan:
	python experiments/scripts/run_paper_experiments.py --dry-run

wheel:
	python -m pip wheel . --no-deps --wheel-dir dist

# 清理
clean:
	@echo "清理 Python 和测试缓存（保留 results/）..."
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
