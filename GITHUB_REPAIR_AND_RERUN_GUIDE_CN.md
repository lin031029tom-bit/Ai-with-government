# GitHub 修复与重新验证流程

## 当前必须修复的文件

1. 用本文件夹中的 `road_safety_dissertation_coding.py` 覆盖 GitHub 根目录同名文件。
2. 用本文件夹中的 `validate_analysis_ready_data.py` 覆盖 GitHub 根目录同名文件。
3. 上传：
   - `.gitignore`
   - `DATA_PREPARATION_NOTES.md`
   - `road_safety_dissertation_coding_clean.ipynb`
4. 用新版 `README.md` 和 `CODING_VALIDATION_REPORT.md` 覆盖旧文件。
5. 保留已经执行过的 `road_safety_dissertation_coding.ipynb`，不要用 clean notebook 覆盖它。

## 为什么必须覆盖主脚本

当前 GitHub 主脚本含有未清理的补丁标记 `@@ -1,442 +1,100 @@`，
并把数据验证代码与建模代码拼接在一起。该版本不能作为最终可运行脚本。

## Colab 重新验证

在全新 Colab 会话中：

```python
!git clone https://github.com/lin031029tom-bit/Ai-with-government.git
%cd /content/Ai-with-government
!pip install -r requirements.txt
```

上传数据到：

```text
/content/Ai-with-government/road_safety_analysis/analysis_ready_road_safety.csv
```

运行验证：

```python
!python validate_analysis_ready_data.py \
  --analysis-ready road_safety_analysis/analysis_ready_road_safety.csv
```

运行主分析：

```python
!python road_safety_dissertation_coding.py \
  --analysis-ready road_safety_analysis/analysis_ready_road_safety.csv \
  --output-dir road_safety_coding_outputs
```

运行全部附加检查：

```python
!python road_safety_dissertation_coding.py \
  --analysis-ready road_safety_analysis/analysis_ready_road_safety.csv \
  --output-dir road_safety_coding_outputs \
  --run-permutation \
  --run-robustness
```

## 本次本地重新运行的主结果

- Balanced logistic regression recall: 0.6419
- Balanced logistic regression ROC-AUC: 0.6613
- Random Forest precision: 0.3540
- Random Forest recall: 0.5920
- Random Forest F1: 0.4430
- Random Forest ROC-AUC: 0.6663
- Random Forest average precision: 0.3933

这些结果与论文 Table 4.3 和 Table 4.4 一致。

## 最终提交前检查

GitHub 根目录应至少包含：

```text
README.md
requirements.txt
.gitignore
CODING_VALIDATION_REPORT.md
DATA_PREPARATION_NOTES.md
validate_analysis_ready_data.py
road_safety_dissertation_coding.py
road_safety_dissertation_coding.ipynb
road_safety_dissertation_coding_clean.ipynb
example_results/
```
