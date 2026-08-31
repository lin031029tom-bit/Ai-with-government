# Windows reproducibility environment

This folder documents a Windows-oriented environment for the dissertation code.
It does not replace the original data, modelling scripts or verified result
files; it adds a fuller dependency lock and a repeatable setup path for reviewers
who run the project on a fresh machine.

## Why this folder exists

The modelling workflow and Random Forest results can be reproduced, but exact
`1e-9` table comparison is sensitive to the Python, NumPy, SciPy, scikit-learn
and low-level numerical stack used by the runner. The root `requirements.txt`
pins the direct packages used by the dissertation, but it does not lock all
transitive dependencies or describe the operating-system environment.

Use this folder when a Windows user needs a more controlled environment than the
root quick-start instructions provide.

## Files

| File | Purpose |
|---|---|
| `requirements-lock-windows.txt` | Full pip-style pinned dependency list, including transitive packages such as SciPy, joblib and threadpoolctl |
| `environment.yml` | Conda environment definition that installs the same pinned Python packages |
| `run_reproduction_windows.ps1` | PowerShell helper that creates a Python 3.12 virtual environment, installs the lock file, runs tests and executes reproduction |

## Recommended Windows run

Open PowerShell in the repository root and run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\reproducibility\windows_environment\run_reproduction_windows.ps1
```

The script performs these steps:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade "pip==25.0.1"
.\.venv\Scripts\python.exe -m pip install -r .\reproducibility\windows_environment\requirements-lock-windows.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe reproduce_dissertation.py
```

The default reproduction remains strict and verifies generated tables using
`rtol=1e-9` and `atol=1e-9`.

## Conda alternative

If Python 3.12 is easier to manage through Conda or Miniconda:

```powershell
conda env create -f .\reproducibility\windows_environment\environment.yml
conda activate road-safety-dissertation-win
python -m unittest discover -s tests -v
python reproduce_dissertation.py
```

## Handling numerical drift

If the full analysis runs and the generated results are substantively the same,
but strict verification fails by tiny floating-point differences, rerun with an
explicitly documented tolerance:

```powershell
python reproduce_dissertation.py --rtol 1e-6 --atol 1e-8
```

This does not replace the strict dissertation benchmark. It is a diagnostic path
for cross-environment validation when different BLAS/OpenMP or wheel builds
produce small numerical differences.

## Environment recording

New runs write expanded environment metadata into
`road_safety_coding_outputs\tables\run_information.json`, including:

- Python version and implementation details;
- Windows/macOS/Linux platform fields;
- direct dependencies from the dissertation;
- transitive numerical dependencies, including SciPy, joblib and threadpoolctl.

This makes future reproduction failures easier to diagnose without changing the
published modelling logic.
