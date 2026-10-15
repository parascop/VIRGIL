# Start Vigil (pipeline + dashboard) on Windows.  Ctrl-C (or closing the window) stops both.
#   powershell -ExecutionPolicy Bypass -File scripts\run_windows.ps1 [-Profile configs\profile_rtx2050.yaml] [-Speed 1] [-NoVlm] [-Loop]
param(
  [string]$Profile = "",
  [double]$Speed = 1.0,
  [switch]$NoVlm,
  [switch]$Loop,
  [int]$Port = 8501
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$py = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }
$env:ANONYMIZED_TELEMETRY = "False"

$pipeArgs = @("-m", "vigil.pipeline", "--speed", "$Speed")
if ($Profile -ne "") { $pipeArgs += @("--profile", $Profile); $env:VIGIL_PROFILE = $Profile }   # otherwise auto-detected (nvidia-smi / VRAM)
if ($NoVlm) { $pipeArgs += "--no-vlm" }
if ($Loop) { $pipeArgs += "--loop" }

$pipe = Start-Process -FilePath $py -ArgumentList $pipeArgs -NoNewWindow -PassThru
$ui = Start-Process -FilePath $py -ArgumentList @("-m", "streamlit", "run", "app/dashboard.py", "--server.port", "$Port", "--server.headless", "true") -NoNewWindow -PassThru
Write-Host "Dashboard: http://localhost:$Port   (pipeline pid $($pipe.Id))"
try {
  while (-not $pipe.HasExited -and -not $ui.HasExited) { Start-Sleep -Seconds 1 }
} finally {
  foreach ($p in @($ui, $pipe)) {
    if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
  }
}
