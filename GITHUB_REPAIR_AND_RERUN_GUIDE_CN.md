# GitHub 验证与重新运行流程

## 当前验证状态

- PR #1 已合并，最终代码位于 `main`。
- 完整数据、模型和结果验证基于提交
  `2148d0950fe026f66dd5ea1810807af32ce91af1`。
- 该提交使用 Python 3.12、503,475 条完整分析记录、1,000 次 bootstrap、
  滚动年份验证、置换重要性和七组稳健性检查完成一键复现。
- 十张关键结果表在 `1e-9` 数值容差下全部匹配。
- 当前自动测试包括语法、数据验证、模型、命令行、一键复现、结果比较和
  notebook 质量检查，共 35 项。

仓库链接：
https://github.com/lin031029tom-bit/Ai-with-government

## 数据要求与边界

经过验证的分析就绪数据已作为 gzip 压缩包发布：

```text
published_data/analysis_ready_road_safety.csv.gz
```

一键复现命令会自动解压到以下忽略路径，无需手动上传 CSV：

```text
road_safety_analysis/analysis_ready_road_safety.csv
```

数据包含 503,475 行。解压后的 SHA-256 为：

```text
5e629f2d931948429580ed778b636b31aa0775630b3e4475727e39df8ee630e1
```

仓库可以精确重复严格数据验证、描述分析、全量建模、bootstrap 置信区间、
滚动年份验证、校准分析、置换重要性和稳健性分析。仓库仍不包含从全部
Department for Transport 原始文件重建分析就绪数据的完整测试流水线，因此
复现边界是模型级精确复现，而不是从每个原始文件独立重建。

## 本地一键重新运行

使用 Python 3.12 创建环境：

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

运行全部自动测试：

```bash
python -m py_compile \
  analysis_schema.py \
  reproduce_dissertation.py \
  road_safety_dissertation_coding.py \
  validate_analysis_ready_data.py \
  verify_dissertation_results.py
python -m unittest discover -s tests -v
```

运行完整论文复现：

```bash
python reproduce_dissertation.py
```

该命令会自动验证环境和数据、运行全部分析，并把关键结果表与经过验证的
参考结果逐值比较。任一阶段失败都会返回非零状态。

## Colab 重新运行

打开顶层 `road_safety_dissertation_coding_clean.ipynb`，或点击 notebook 顶部
的 **Open in Colab** 按钮，然后选择“运行所有单元格”。clean notebook 会：

1. 检出包含完整数据和结果的已验证提交 `2148d095...`；
2. 安装锁定版本的依赖；
3. 验证发布压缩包的 SHA-256；
4. 运行 `python reproduce_dissertation.py`；
5. 展示生成的表格和图片。

旧的失败执行记录已移动到 `archive/`，只保留为历史来源记录，不应作为最终
代码运行状态的证据。

## 已验证的主要结果

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Average precision | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dummy prevalence baseline | 0.7516 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2484 | 0.1870 |
| Balanced logistic regression | 0.6977 | 0.3885 | 0.3782 | 0.3833 | 0.6519 | 0.3829 | 0.2012 |
| Random Forest | 0.6273 | 0.3588 | 0.6355 | 0.4586 | 0.6877 | 0.4227 | 0.2211 |

Random Forest 在 2021–2024 四个滚动测试年的 ROC-AUC 为 0.6937、0.6958、
0.7012 和 0.6877；2024 结果的 95% bootstrap 区间为 0.6841–0.6915。
完整来源信息见 `CODING_VALIDATION_REPORT.md`、`published_results/` 和
`published_data/`。

## 提交或分享前检查

- `git status` 没有未提交修改；
- GitHub Actions 在最新 `main` 提交上成功；
- clean notebook 没有保存的错误输出，并固定到已验证的数据发布快照；
- `CODING_VALIDATION_REPORT.md` 与当前测试数和验证提交一致；
- `published_data/` 的哈希与报告、论文一致；
- `published_results/` 与论文报告的结果一致；
- 不提交虚拟环境、缓存、解压后的工作 CSV 或临时输出。
