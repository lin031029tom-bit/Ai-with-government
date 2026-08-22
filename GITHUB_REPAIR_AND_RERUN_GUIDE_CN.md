# GitHub 验证与重新运行流程

## 当前验证状态

- 修改位于 Draft PR #1 的 `codex/validation-ci` 分支，审核完成前不要合并到 `main`。
- `edaa214` 是使用完整分析数据执行全量训练、预测误差验证并通过结果核验的代码提交。
- 后续产物提交记录完整数据验证报告和示例输出，不改变已验证的建模代码。
- 当前仓库测试共 29 项；提交后以 PR 的 Checks 页面为最终 GitHub Actions 依据。
- 验证环境为 Python 3.12；仓库根目录的 `.python-version` 记录了该版本。

PR 链接：
https://github.com/lin031029tom-bit/Ai-with-government/pull/1

## 数据要求与边界

分析需要以下文件，但由于访问限制和文件大小，该文件不随仓库分发：

```text
road_safety_analysis/analysis_ready_road_safety.csv
```

已验证文件包含 503,475 行，SHA-256 为：

```text
5e629f2d931948429580ed778b636b31aa0775630b3e4475727e39df8ee630e1
```

仓库可以从该分析就绪文件开始，重复数据验证、全量建模、bootstrap 置信区间、
滚动年份验证、校准分析、置换检验和稳健性分析。
仓库不包含从 Department for Transport 原始文件重建完整分析数据集的端到端流水线，
因此不能声称仅凭仓库即可从原始数据复现全部数据准备过程。

## 本地重新运行

使用 Python 3.12 创建独立环境：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows 可用：

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

验证数据：

```bash
python validate_analysis_ready_data.py \
  --analysis-ready road_safety_analysis/analysis_ready_road_safety.csv
```

推荐的一键完整复现：

```bash
python reproduce_dissertation.py \
  --analysis-ready road_safety_analysis/analysis_ready_road_safety.csv \
  --output-dir road_safety_coding_outputs
```

该命令会依次运行严格数据验证、全量主分析、1,000 次成对 bootstrap、2021-2024
滚动年份验证、置换重要性和稳健性分析，并把十张关键结果表与仓库中经过验证的结果
逐值比较。任一阶段失败时命令都会返回失败状态。

如需单独运行主脚本：

```bash
python road_safety_dissertation_coding.py \
  --analysis-ready road_safety_analysis/analysis_ready_road_safety.csv \
  --output-dir road_safety_coding_outputs \
  --full-training \
  --bootstrap-iterations 1000 \
  --run-temporal-validation \
  --run-permutation \
  --run-robustness
```

不需要受限数据的自动测试：

```bash
python -m unittest discover -s tests -v
```

## Colab 重新运行

推荐打开 `road_safety_dissertation_coding_clean.ipynb`。该 notebook 固定到已验证的
`edaa214` 提交，避免 `main` 或 PR 后续变化造成结果漂移。按 notebook 提示上传分析就绪
CSV，然后运行所有单元格。

`road_safety_dissertation_coding.ipynb` 是保留的 Colab 执行记录，包括设置和上传数据时的
排错过程；不要用 clean notebook 覆盖它。

## 已验证的主要结果

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Average precision | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dummy prevalence baseline | 0.7516 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2484 | 0.1870 |
| Balanced logistic regression | 0.6977 | 0.3885 | 0.3782 | 0.3833 | 0.6519 | 0.3829 | 0.2012 |
| Random Forest | 0.6273 | 0.3588 | 0.6355 | 0.4586 | 0.6877 | 0.4227 | 0.2211 |

Random Forest 在 2021-2024 四个滚动测试年的 ROC-AUC 为 0.6937、0.6958、0.7012
和 0.6877；2024 结果的 95% bootstrap 区间为 0.6841-0.6915。完整数值、校准图、
置信区间和运行来源见 `CODING_VALIDATION_REPORT.md` 与 `example_results/`。

## 审核清单

合并 PR 前确认：

- PR 仍为 Draft，且没有直接修改 `main`。
- GitHub Actions 在 PR 最新提交上通过。
- Python 版本为 3.12，依赖来自 `requirements.txt`。
- `analysis_schema.py`、验证脚本、主脚本、一键复现脚本与 `tests/` 均存在。
- 两个 notebook 均通过格式验证，clean notebook 仍固定到 `edaa214`。
- `CODING_VALIDATION_REPORT.md`、`DATA_PREPARATION_NOTES.md` 和
  `example_results/` 与论文中报告的结果一致。
- 不提交受限 CSV、虚拟环境、缓存或临时输出。
