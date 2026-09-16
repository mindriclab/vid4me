# Modellgewichte von Hugging Face herunterladen.
# Nutzung:
#   scripts\download_models.ps1              -> fragt interaktiv
#   scripts\download_models.ps1 chroma       -> nur Chroma1-HD (~18 GB)
#   scripts\download_models.ps1 i2v          -> Wan 2.2 I2V A14B (~55 GB)
#   scripts\download_models.ps1 t2v          -> Wan 2.2 T2V A14B (~55 GB)
#   scripts\download_models.ps1 all          -> alles
param([string]$Target = "")

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Error "Bitte zuerst scripts\setup_backend.ps1 ausfuehren."; exit 1 }

$models = @{
    "chroma" = @{ repo = "lodestones/Chroma1-HD";              dest = "models\image\chroma" }
    "i2v"    = @{ repo = "Wan-AI/Wan2.2-I2V-A14B-Diffusers";   dest = "models\video\wan22-14b\i2v" }
    "t2v"    = @{ repo = "Wan-AI/Wan2.2-T2V-A14B-Diffusers";   dest = "models\video\wan22-14b\t2v" }
}

if ($Target -eq "") {
    Write-Host "Welche Modelle herunterladen? (chroma / i2v / t2v / all)" -ForegroundColor Cyan
    $Target = Read-Host "Auswahl"
}

$keys = if ($Target -eq "all") { @("chroma", "i2v", "t2v") } else { @($Target) }

foreach ($key in $keys) {
    if (-not $models.ContainsKey($key)) { Write-Warning "Unbekanntes Ziel: $key"; continue }
    $m = $models[$key]
    $dest = Join-Path $root $m.dest
    Write-Host "Lade $($m.repo) nach $dest ..." -ForegroundColor Cyan
    & $py -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$($m.repo)', local_dir=r'$dest')"
}
Write-Host "Fertig." -ForegroundColor Green
