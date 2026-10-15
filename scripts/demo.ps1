# One-command demo (Windows): make synthetic clips if missing, start the looping pipeline, open the dashboard.
#   powershell -ExecutionPolicy Bypass -File scripts\demo.ps1 [-NoVlm]
param([switch]$NoVlm)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$py = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }
$env:ANONYMIZED_TELEMETRY = "False"
if (-not (Test-Path "data\clips\demo_cam1.mp4")) { & $py scripts\make_demo_clips.py }
& $py scripts\diagnose.py --skip-streamlit
$pipeArgs = @("-m", "vigil.pipeline", "--speed", "4", "--loop")
if ($NoVlm) { $pipeArgs += "--no-vlm" }
$pipe = Start-Process -FilePath $py -ArgumentList $pipeArgs -NoNewWindow -PassThru
try {
  Start-Sleep -Seconds 5
  & $py -m streamlit run app\dashboard.py --server.headless false
} finally {
  if ($pipe -and -not $pipe.HasExited) { Stop-Process -Id $pipe.Id -Force -ErrorAction SilentlyContinue }
}
