param(
    [string]$Rtol = "1e-9",
    [string]$Atol = "1e-9"
)

$ErrorActionPreference = "Stop"

$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepositoryRoot = Resolve-Path (Join-Path $ScriptDirectory "..\..")
$PythonExe = Join-Path $RepositoryRoot ".venv\Scripts\python.exe"

Set-Location $RepositoryRoot

if (-not (Test-Path $PythonExe)) {
    py -3.12 -m venv .venv
}

& $PythonExe -m pip install --upgrade "pip==25.0.1"
& $PythonExe -m pip install -r ".\reproducibility\windows_environment\requirements-lock-windows.txt"
& $PythonExe -m unittest discover -s tests -v
& $PythonExe reproduce_dissertation.py --rtol $Rtol --atol $Atol
