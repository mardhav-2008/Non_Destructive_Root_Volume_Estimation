
$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host " NON-DESTRUCTIVE ROOT VOLUME ESTIMATION"
Write-Host " Windows 11 - CPU Setup"
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

Set-Location $PSScriptRoot

# --------------------------------------------------
# 1. Check Python
# --------------------------------------------------

Write-Host "[1/6] Checking Python..." -ForegroundColor Yellow

$PythonLauncher = Get-Command py -ErrorAction SilentlyContinue

if (-not $PythonLauncher) {
    Write-Host "Python launcher not found." -ForegroundColor Red
    Write-Host "Install Python 3.12 or 3.13 from python.org."
    exit 1
}

$PythonVersion = $null

foreach ($Version in @("3.12", "3.13")) {
    try {
        & py "-$Version" --version 2>$null

        if ($LASTEXITCODE -eq 0) {
            $PythonVersion = $Version
            break
        }
    }
    catch {
        continue
    }
}

if (-not $PythonVersion) {
    Write-Host "Python 3.12 or 3.13 is required." -ForegroundColor Red
    exit 1
}

Write-Host "Using Python $PythonVersion" -ForegroundColor Green

# --------------------------------------------------
# 2. Check project files
# --------------------------------------------------

Write-Host ""
Write-Host "[2/6] Checking project files..." -ForegroundColor Yellow

$RequiredFiles = @(
    "requirements.txt",
    "scripts\gui.py",
    "src\preprocessing.py",
    "src\dataset.py",
    "src\models\cnn_regressor.py",
    "Test.csv",
    "data\test",
    "outputs\uncertainty\image_fixed_epoch.pt",
    "outputs\uncertainty\image_metrics.json"
)

$Missing = @()

foreach ($File in $RequiredFiles) {
    if (-not (Test-Path $File)) {
        $Missing += $File
    }
}

if ($Missing.Count -gt 0) {
    Write-Host "Missing required project files:" -ForegroundColor Red

    foreach ($File in $Missing) {
        Write-Host "  - $File" -ForegroundColor Red
    }

    Write-Host ""
    Write-Host "Copy the complete project and trained model files."
    exit 1
}

Write-Host "Project files verified." -ForegroundColor Green

# --------------------------------------------------
# 3. Create virtual environment
# --------------------------------------------------

Write-Host ""
Write-Host "[3/6] Creating virtual environment..." -ForegroundColor Yellow

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    & py "-$PythonVersion" -m venv .venv
}

$VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

# --------------------------------------------------
# 4. Install CPU PyTorch
# --------------------------------------------------

Write-Host ""
Write-Host "[4/6] Installing CPU-only PyTorch..." -ForegroundColor Yellow

& $VenvPython -m pip install --upgrade pip

if ($LASTEXITCODE -ne 0) {
    throw "Could not upgrade pip."
}

& $VenvPython -m pip install torch --index-url https://download.pytorch.org/whl/cpu

if ($LASTEXITCODE -ne 0) {
    throw "PyTorch installation failed."
}

# --------------------------------------------------
# 5. Install remaining dependencies
# --------------------------------------------------

Write-Host ""
Write-Host "[5/6] Installing project dependencies..." -ForegroundColor Yellow

& $VenvPython -m pip install -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    throw "Dependency installation failed."
}

# --------------------------------------------------
# 6. Verify and launch
# --------------------------------------------------

Write-Host ""
Write-Host "[6/6] Verifying installation..." -ForegroundColor Yellow

& $VenvPython -c "import torch, pandas, numpy, sklearn, PIL; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CPU:', torch.get_num_threads())"

if ($LASTEXITCODE -ne 0) {
    throw "Dependency verification failed."
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host " INSTALLATION COMPLETE" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""

Write-Host "Launching Root Volume Estimator..." -ForegroundColor Cyan

& $VenvPython scripts\gui.py
