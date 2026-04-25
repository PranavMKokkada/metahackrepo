<#!
  One-shot local baseline for Windows + Python 3.11:
  - Ensures .venv311 exists
  - Installs requirements.txt + requirements-training-win311.txt + matplotlib
  - Runs pytest, policy evaluation, plots, and evaluation report
  - Optionally runs GRPO recipe mode (no GPU)

  Usage (from repo root, PowerShell):
    .\scripts\run_local_baseline_win311.ps1
    .\scripts\run_local_baseline_win311.ps1 -SkipGrpoRecipe
#>
param(
    [switch]$SkipGrpoRecipe
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$Py311 = "py"
$VenvPython = Join-Path $RepoRoot ".venv311\Scripts\python.exe"
$VenvPip = Join-Path $RepoRoot ".venv311\Scripts\pip.exe"

Write-Host "==> Repo root: $RepoRoot"

Write-Host "==> Creating venv with Python 3.11 (.venv311) if missing..."
if (-not (Test-Path $VenvPython)) {
    & $Py311 -3.11 -m venv (Join-Path $RepoRoot ".venv311")
}

Write-Host "==> Upgrading pip..."
& $VenvPython -m pip install --upgrade pip

Write-Host "==> Installing runtime requirements..."
& $VenvPython -m pip install -r (Join-Path $RepoRoot "requirements.txt")

Write-Host "==> Installing Windows-safe training stack (no Unsloth/xformers)..."
& $VenvPython -m pip install -r (Join-Path $RepoRoot "requirements-training-win311.txt")

Write-Host "==> Installing matplotlib (for plot_results.py)..."
& $VenvPython -m pip install matplotlib

Write-Host "==> Running pytest..."
& $VenvPython -m pytest -q

Write-Host "==> Running policy evaluation..."
& $VenvPython (Join-Path $RepoRoot "training\evaluate_policy.py") `
    --policies noop random heuristic stabilized sft `
    --episodes-per-phase 6 `
    --out-dir (Join-Path $RepoRoot "results")

Write-Host "==> Generating plots..."
& $VenvPython (Join-Path $RepoRoot "training\plot_results.py") `
    --results-dir (Join-Path $RepoRoot "results") `
    --summary (Join-Path $RepoRoot "results\eval_summary.json")

Write-Host "==> Writing evaluation report..."
& $VenvPython (Join-Path $RepoRoot "evaluation\run_eval.py")

if (-not $SkipGrpoRecipe) {
    Write-Host "==> GRPO recipe mode (no GPU, generates HF job recipe + notebook metrics extract)..."
    & $VenvPython (Join-Path $RepoRoot "training\grpo_train.py") --mode grpo
}

Write-Host ""
Write-Host "Done. Key outputs:"
Write-Host "  - results\eval_summary.json"
Write-Host "  - results\*.png"
Write-Host "  - evaluation\report.md"
Write-Host "  - results\grpo_gpu_recipe.json (if GRPO recipe ran)"
Write-Host ""
