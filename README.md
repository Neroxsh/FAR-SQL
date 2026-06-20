# FAR-SQL

FAR-SQL 是一个面向 SQL 等价性判定的失效感知分层框架：先保留 VeriEQL 能够形式化判定的结果，再把 `timeout`、`unsupported`、`runtime_error`、`conversion_error` 等非确定状态交给本地 SQL 等价性 SFT 模型补判。

本仓库整理了论文相关的代码、标准化测试数据、冻结实验产物和最终论文稿件。仓库刻意只保留可复现、可审计、可说明的内容；早期多方向尝试产生的大日志、模型权重、临时结果和错误口径实验未纳入版本库。

## 结果口径

仓库保留两个口径，避免把论文提交结果和更严格的方法定义混在一起：

| 口径 | 含义 |
|---|---|
| `strict-farsql` | 严格按方法描述执行：VeriEQL 输出 `equivalent`/`non_equivalent` 时直接采用；所有非确定状态全部交给本地模型做 Self-Consistency@3。 |
| `paper-reproduction` | 复现提交论文中使用的冻结结果。它同样保持形式化优先，但包含 85 条 timeout-positive-prior 决策；审计脚本会把这部分单独统计为 `Prior-only`。 |

只审计冻结结果不需要 GPU、模型权重或 VeriEQL：

```bash
PYTHONPATH=src python src/verify_paper_artifacts.py
```

关键审计行应包含：

```text
paper-reproduction  Qwen2.5-SFT  GM 83.98  Prior-only 85
paper-reproduction  Qwen3-SFT    GM 83.69  Prior-only 85
strict-farsql       Qwen2.5-SFT  GM 80.14  Prior-only 0
strict-farsql       Qwen3-SFT    GM 80.53  Prior-only 0
```

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python -m unittest discover -s tests
PYTHONPATH=src python src/verify_paper_artifacts.py
```

完整重跑实验需要额外准备 VeriEQL、原始数据、本地 SFT checkpoint 和 GPU/vLLM 环境。路径可在 `configs/default.yaml` 中修改。

## 目录和文件说明

### 根目录

| 文件/目录 | 说明 |
|---|---|
| `.gitignore` | 忽略 Python 缓存、虚拟环境、运行输出、原始数据、模型权重、第三方工具和临时日志。 |
| `CITATION.cff` | GitHub 识别用引用信息，说明如何引用 FAR-SQL artifact。 |
| `LICENSE` | MIT 开源许可证。 |
| `README.md` | 本文件，说明仓库用途、结果口径、复现命令和所有文件含义。 |
| `configs/` | 实验配置目录。 |
| `data/` | 标准化后的轻量测试数据目录；不包含原始私有数据。 |
| `docs/` | 方法、复现、产物和历史整理说明。 |
| `paper/` | 最终论文稿件目录。 |
| `paper_artifacts/` | 论文结果审计所需的冻结决策、指标和表格。 |
| `requirements.txt` | CPU/审计路径需要的 Python 依赖。 |
| `requirements-gpu.txt` | GPU 推理路径需要的额外依赖。 |
| `scripts/` | 从数据准备到实验评估的命令行入口。 |
| `src/` | FAR-SQL 数据处理、路由、推理、融合和评估源码。 |
| `tests/` | 当前保留的小型回归测试。 |

### `configs/`

| 文件 | 说明 |
|---|---|
| `configs/default.yaml` | 默认实验配置，包含项目根目录、VeriEQL 路径占位、原始数据占位、标准化数据路径、候选预算、模型键名、推理参数和实验输出目录。 |

### `data/standard/`

| 文件 | 说明 |
|---|---|
| `data/standard/dataset_manifest.json` | 标准化数据清单，记录每个 split 的文件路径、样本数和来源摘要。 |
| `data/standard/test_calcite_spider.jsonl` | 论文主要测试集，412 条样本，由 Calcite 等价样本和 Spider-DAIL 非等价样本组成。 |
| `data/standard/test_leetcode.jsonl` | LeetCode 测试集，1300 条标准化样本，用于补充实验和模型行为检查。 |

### `docs/`

| 文件 | 说明 |
|---|---|
| `docs/ARCHIVE_NOTES.md` | 说明早期探索方向和大体归档策略，避免读者误以为未提交的大日志仍是论文主线。 |
| `docs/ARTIFACTS.md` | 说明 `paper_artifacts/` 中冻结产物的来源、用途和注意事项。 |
| `docs/METHOD.md` | FAR-SQL 方法口径说明，强调形式化确定区与非确定补判区的边界。 |
| `docs/REPRODUCIBILITY.md` | 复现实验说明，包括冻结产物审计、CPU/形式化阶段和完整 GPU 重跑所需条件。 |

### `paper/`

| 文件 | 说明 |
|---|---|
| `paper/面向 SQL 等价性判定的失效感知分层方法.docx` | 最终提交论文的 Word 版本。 |
| `paper/面向 SQL 等价性判定的失效感知分层方法.pdf` | 最终提交论文的 PDF 版本，便于直接阅读和归档。 |

### `paper_artifacts/decisions/`

| 文件 | 说明 |
|---|---|
| `paper_artifacts/decisions/test_calcite_spider.fixed10s.verieql_stage.jsonl` | Calcite+Spider 测试集在 10 秒固定 VeriEQL 预算下的形式化阶段冻结结果。 |
| `paper_artifacts/decisions/test_calcite_spider.fixed30s.verieql_stage.jsonl` | Calcite+Spider 测试集在 30 秒固定 VeriEQL 预算下的形式化阶段冻结结果。 |
| `paper_artifacts/decisions/test_calcite_spider.fixed60s.verieql_stage.jsonl` | Calcite+Spider 测试集在 60 秒固定 VeriEQL 预算下的形式化阶段冻结结果。 |
| `paper_artifacts/decisions/test_calcite_spider.fixed120s.verieql_stage.jsonl` | Calcite+Spider 测试集在 120 秒固定 VeriEQL 预算下的形式化阶段冻结结果。 |
| `paper_artifacts/decisions/test_calcite_spider.adaptive_accuracy.verieql_stage.jsonl` | Calcite+Spider 测试集在动态预算策略 `adaptive_accuracy` 下的 VeriEQL 形式化阶段冻结结果。 |
| `paper_artifacts/decisions/test_calcite_spider.adaptive_accuracy.qwen25_coder_1_5b_100sft_cot.qwen25_sft_clean_timeout_llm_verify.final.jsonl` | `strict-farsql` 口径下，Qwen2.5-SFT 的最终融合决策；非确定状态由 LLM 补判，`Prior-only=0`。 |
| `paper_artifacts/decisions/test_calcite_spider.adaptive_accuracy.qwen3_1_7b_100sft_cot.qwen3_sft_clean_timeout_llm_verify.final.jsonl` | `strict-farsql` 口径下，Qwen3-SFT 的最终融合决策；非确定状态由 LLM 补判，`Prior-only=0`。 |
| `paper_artifacts/decisions/test_calcite_spider.adaptive_accuracy.qwen25_coder_1_5b_100sft_cot.failure_mode_routed.final.jsonl` | `paper-reproduction` 口径下，Qwen2.5-SFT 的提交论文复现决策；包含 85 条 prior-only 决策。 |
| `paper_artifacts/decisions/test_calcite_spider.adaptive_accuracy.qwen3_1_7b_100sft_cot.failure_mode_routed.final.jsonl` | `paper-reproduction` 口径下，Qwen3-SFT 的提交论文复现决策；包含 85 条 prior-only 决策。 |

### `paper_artifacts/metrics/`

| 文件 | 说明 |
|---|---|
| `paper_artifacts/metrics/failure_mode_routed_calcite_spider.metrics.json` | 提交论文复现口径的 Calcite+Spider 指标汇总和路由摘要。 |
| `paper_artifacts/metrics/qwen25_sft_clean_timeout_llm_verify.metrics.json` | Qwen2.5-SFT 严格补判口径的指标汇总。 |
| `paper_artifacts/metrics/qwen3_sft_clean_timeout_llm_verify.metrics.json` | Qwen3-SFT 严格补判口径的指标汇总。 |

### `paper_artifacts/tables/`

| 文件 | 说明 |
|---|---|
| `paper_artifacts/tables/submitted_table1_main.csv` | 提交论文主结果表的冻结 CSV。 |
| `paper_artifacts/tables/submitted_table2_ablation.csv` | 提交论文消融实验表的冻结 CSV。 |
| `paper_artifacts/tables/submitted_table3_efficiency.csv` | 提交论文效率对比表的冻结 CSV。 |

### `scripts/`

| 文件 | 说明 |
|---|---|
| `scripts/env.sh` | 所有脚本共用的环境初始化，设置 `NDBC_ROOT`、`PYTHONPATH` 和 Python 解释器。 |
| `scripts/00_preflight.sh` | 运行配置、数据、模型路径和答案解析等预检查。 |
| `scripts/00_prepare_data.sh` | 根据配置把原始测试数据标准化为 `data/standard/*.jsonl`。 |
| `scripts/01_calibrate_budget.sh` | 从校准日志中统计不同预算的覆盖率和预算表。 |
| `scripts/01b_build_calibration_audit_and_routing.sh` | 学习/汇总动态预算路由策略，并生成校准审计报告。 |
| `scripts/02_run_verieql_budgeted_test.sh` | 按动态预算策略运行测试集 VeriEQL 阶段。 |
| `scripts/03_build_fallback_prompts.sh` | 为 VeriEQL 非确定样本构造 LLM fallback prompt。 |
| `scripts/04_run_fallback_gpu.sh` | 使用 vLLM/Transformers 跑本地模型 fallback 推理。 |
| `scripts/05_fuse_decisions.sh` | 将 VeriEQL 形式化结果和 LLM fallback 日志融合成最终判定。 |
| `scripts/07_build_model_only_prompts.sh` | 构造不使用 VeriEQL 的纯模型 baseline prompt。 |
| `scripts/08_run_model_only_gpu.sh` | 运行纯模型 baseline 推理。 |
| `scripts/09_compute_all_experiments.sh` | 汇总实验输出并生成论文表格。 |
| `scripts/10_run_verieql_fixed_budgets.sh` | 运行固定预算 VeriEQL 对比，默认覆盖 1/2/3/5/10/30/60/120 秒。 |
| `scripts/11_run_verieql_official_600.sh` | 运行 600 秒 VeriEQL 官方长预算参考。 |
| `scripts/12_adaptive_offline_search.sh` | 离线搜索动态预算策略，用于比较不同策略收益。 |
| `scripts/13_run_verieql_aware_fallback.sh` | 运行带 VeriEQL 状态提示的 fallback prompt、推理和融合流程。 |
| `scripts/14_run_qwen3_base_suite.sh` | 运行 Qwen3-Base 相关的 prompt/路由实验套件。 |
| `scripts/15_repeat_clean_runs.sh` | 对指定模型和路由标签重复运行干净版实验，便于检查随机性。 |
| `scripts/99_clean_outputs.sh` | 清理 `outputs/` 和 Python 缓存，并重建输出目录骨架。 |
| `scripts/run_cpu_smoke.sh` | 小样本 CPU smoke test，用于快速验证配置和非 GPU 流水线。 |

### `src/`

| 文件 | 说明 |
|---|---|
| `src/adaptive_offline_search.py` | 离线学习并评估动态预算策略，支持生成动态 VeriEQL stage 和策略报告。 |
| `src/analyze_calibration_buckets.py` | 分析校准集在不同特征桶和预算下的覆盖率、运行时间和预算选择。 |
| `src/build_fallback_prompts.py` | 为非确定 VeriEQL 样本构造多种 fallback prompt，包括极简提示、VeriEQL-aware、trace/witness/counterexample 引导等。 |
| `src/build_model_only_prompts.py` | 构造纯 LLM baseline 使用的 prompt。 |
| `src/calibrate_budget.py` | 从历史 VeriEQL 校准记录中归一化状态、统计覆盖率并写出预算表。 |
| `src/common.py` | 公共工具函数：配置读取、路径解析、JSONL 读写、标签归一化、哈希、均值/分位数和标准路径管理。 |
| `src/compute_all_experiment_tables.py` | 旧版总表计算入口的保留文件；当前提交论文表格以 `compute_paper_tables.py` 和冻结 CSV 为准。 |
| `src/compute_paper_tables.py` | 汇总形式化阶段、模型 baseline、融合结果并生成论文表格。 |
| `src/convert_verieql_ready.py` | 将标准化数据转换成 VeriEQL 可消费的 schema/query 格式。 |
| `src/evaluate_experiment_suite.py` | 对形式化-only、模型-only、hybrid 和预计算 final 结果做统一评估。 |
| `src/extract_features.py` | 从 SQL 对和 schema 中抽取长度、嵌套、join、聚合、约束等路由特征。 |
| `src/fuse_decisions.py` | 实现形式化优先融合：VeriEQL 已判定时采用形式化结果，非确定样本读取 LLM 日志补判。 |
| `src/learn_routing_policy.py` | 根据校准结果学习动态预算路由策略，并输出人工可读报告。 |
| `src/normalize_verieql.py` | 将 VeriEQL 原始输出归一化为统一状态、运行时间和最终标签字段。 |
| `src/preflight_checks.py` | 检查配置文件、输入数据、模型路径、转换依赖和答案解析逻辑。 |
| `src/prepare_data.py` | 将原始 LeetCode/Calcite/Spider-DAIL 数据转换成统一 JSONL 样本格式。 |
| `src/routing_policy.py` | 动态预算策略的核心函数，包括特征分桶、策略学习、路由选择和路由摘要。 |
| `src/run_fallback_gpu.py` | 使用本地模型对 prompt 批量生成答案，支持多采样和 GPU 推理参数。 |
| `src/run_status_aware_router.py` | 状态感知路由融合实验入口，支持严格 LLM 补判和提交论文复现口径。 |
| `src/run_status_specialist.py` | 失效状态 specialist/variant 对比实验入口，用于分析不同提示策略在不同状态下的表现。 |
| `src/run_verieql_budgeted.py` | 调用 VeriEQL，在固定预算或动态预算下生成形式化阶段结果。 |
| `src/summarize_autobudget_routing.py` | 汇总动态预算路由计划，输出人读 Markdown 报告。 |
| `src/summarize_model_suite.py` | 汇总模型-only、fallback 和 router 结果，便于比较同一模型的多种设置。 |
| `src/verieql_simulation.py` | 根据历史 VeriEQL 轨迹模拟不同预算下的终止状态和耗时。 |
| `src/verify_paper_artifacts.py` | 冻结产物审计脚本，重新计算 Acc_eq、Acc_neq、GM、Overall、Und、Avg/P95 latency 和 Prior-only 数量。 |

### `tests/`

| 文件 | 说明 |
|---|---|
| `tests/test_verify_paper_artifacts.py` | 对审计脚本的指标计算、GM 边界情况和 prior-only 计数做回归测试。 |

## 完整复现流程

完整复现需要外部资产，这些资产不纳入仓库：

- VeriEQL 及其 SQL 格式转换器；
- LeetCode、Calcite、Spider-DAIL 原始输入；
- Qwen2.5-SFT 和 Qwen3-SFT 本地 checkpoint；
- 支持 vLLM 的 GPU 环境。

准备好外部资产后，可按以下顺序运行：

```bash
bash scripts/00_preflight.sh
bash scripts/00_prepare_data.sh
bash scripts/01_calibrate_budget.sh
bash scripts/01b_build_calibration_audit_and_routing.sh
bash scripts/02_run_verieql_budgeted_test.sh
bash scripts/10_run_verieql_fixed_budgets.sh
bash scripts/03_build_fallback_prompts.sh
bash scripts/04_run_fallback_gpu.sh
bash scripts/05_fuse_decisions.sh
bash scripts/09_compute_all_experiments.sh
```

更多细节见 `docs/REPRODUCIBILITY.md`。

## 引用

如果本仓库对你的研究有帮助，请引用随仓库提供的论文和 artifact：

```bibtex
@misc{farsql2026,
  title = {Failure-Aware Routing for SQL Equivalence Judgment},
  author = {Xing Shihao},
  year = {2026},
  note = {FAR-SQL artifact}
}
```
