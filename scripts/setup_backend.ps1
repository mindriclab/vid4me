# Vid4me Backend-Setup: Python-Umgebung (.venv) + API-Deps + ML-Stack (torch CUDA fuer RTX-Karten)
#
# Entwicklung:   powershell -ExecutionPolicy Bypass -File scripts\setup_backend.ps1
# Gepackte App:  wird beim ersten Start automatisch mit -DataRoot <Datenordner> aufgerufen
#
# Findet kein Python (>=3.10) auf dem System, wird automatisch ein portables
# Python 3.12 in den Datenordner geladen - die App ist damit selbstversorgend.
param([string]$DataRoot = "")

$ErrorActionPreference = "Stop"
$appRoot = Split-Path -Parent $PSScriptRoot   # Projektordner (dev) bzw. resources-Ordner (gepackt)
if ($DataRoot -eq "") { $DataRoot = $appRoot }
$venvDir = Join-Path $DataRoot ".venv"

Write-Host "=== Vid4me Setup ===" -ForegroundColor Cyan
Write-Host "App-Dateien:  $appRoot"
Write-Host "Datenordner:  $DataRoot"
Write-Host ""

function Find-Python {
    foreach ($cand in @("python", "py")) {
        try {
            $exe = (Get-Command $cand -ErrorAction Stop).Source
            $ok = & $exe -c "import sys; print(1 if sys.version_info >= (3,10) else 0)"
            if ("$ok".Trim() -eq "1") { return $exe }
        } catch { }
    }
    # Bereits heruntergeladenes portables Python?
    $portable = Join-Path $DataRoot "python\tools\python.exe"
    if (Test-Path $portable) { return $portable }
    return $null
}

$python = Find-Python
if (-not $python) {
    Write-Host "Kein Python >= 3.10 gefunden - lade portables Python 3.12 (~30 MB)..." -ForegroundColor Yellow
    $pyDir = Join-Path $DataRoot "python"
    New-Item -ItemType Directory -Force $pyDir | Out-Null
    $zip = Join-Path $env:TEMP "vid4me-python-3.12.9.zip"
    Invoke-WebRequest "https://api.nuget.org/v3-flatcontainer/python/3.12.9/python.3.12.9.nupkg" -OutFile $zip
    Expand-Archive $zip -DestinationPath $pyDir -Force
    Remove-Item $zip -Force
    $python = Join-Path $pyDir "tools\python.exe"
    if (-not (Test-Path $python)) { Write-Error "Portables Python konnte nicht eingerichtet werden."; exit 1 }
}
Write-Host "Python: $python" -ForegroundColor Green

if (-not (Test-Path (Join-Path $venvDir "Scripts\python.exe"))) {
    Write-Host "Erstelle virtuelles Environment ($venvDir)..." -ForegroundColor Cyan
    & $python -m venv $venvDir
}
$py = Join-Path $venvDir "Scripts\python.exe"

Write-Host "Installiere API-Abhaengigkeiten..." -ForegroundColor Cyan
& $py -m pip install --upgrade pip
& $py -m pip install -r (Join-Path $appRoot "backend\requirements.txt")

Write-Host "Installiere PyTorch (CUDA 12.8)... (~3 GB Download, dauert einige Minuten)" -ForegroundColor Cyan
& $py -m pip install torch --index-url https://download.pytorch.org/whl/cu128

Write-Host "Installiere Diffusers-Stack..." -ForegroundColor Cyan
& $py -m pip install "diffusers>=0.38" "transformers>=4.46" "accelerate>=1.0" `
    safetensors "peft>=0.13" sentencepiece ftfy "imageio[ffmpeg]" pillow huggingface_hub `
    torchao hf_transfer av

Write-Host ""
Write-Host "Setup abgeschlossen. CUDA-Check:" -ForegroundColor Green
& $py -c "import torch; print('torch', torch.__version__, '| CUDA verfuegbar:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'keine GPU')"
Write-Host ""
Write-Host "Modelle koennen jetzt direkt in der App heruntergeladen werden (Menue 'Modelle')." -ForegroundColor Yellow
