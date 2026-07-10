# =====================================================================
#  Vision-grinder toggle — Qwen3-VL-4B on the GPU (port 8081).
#
#  Running  -> stops it (frees ~3.8 GB VRAM for games).
#  Stopped  -> starts it (manga pages ~15 s each; photo memory grind).
#
#  Independent of the main T-800 stack: the bot works without it — vision
#  batch jobs just fall back to the slow-but-polite 12B path automatically.
#  Double-click `toggle-grinder.cmd` to run this.
# =====================================================================

$ErrorActionPreference = 'SilentlyContinue'

$Exe   = 'D:\llamacpp-cuda\llama-server.exe'
$Model = 'D:\vlm-grinders\qwen3-vl-4b\Qwen3VL-4B-Instruct-Q4_K_M.gguf'
$Proj  = 'D:\vlm-grinders\qwen3-vl-4b\mmproj-Qwen3VL-4B-Instruct-F16.gguf'
$Log   = 'D:\llamacpp-cuda\grinder.log'

function Get-Grinder {
    Get-CimInstance Win32_Process -Filter "Name='llama-server.exe'" |
        Where-Object { $_.CommandLine -like '*8081*' }
}

if (Get-Grinder) {
    Write-Host ''
    Write-Host '  Stopping vision grinder...' -ForegroundColor Yellow
    Get-Grinder | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
    Write-Host '  Grinder OFFLINE. ~3.8 GB VRAM freed.' -ForegroundColor Green
    Write-Host ''
}
else {
    Write-Host ''
    Write-Host '  Starting vision grinder (Qwen3-VL-4B, GPU)...' -ForegroundColor Yellow
    Start-Process -FilePath $Exe -ArgumentList @(
        '-m', $Model, '--mmproj', $Proj,
        '-ngl', '99', '-c', '2560', '-t', '4', '--jinja',
        '--host', '127.0.0.1', '--port', '8081'
    ) -WorkingDirectory 'D:\llamacpp-cuda' -WindowStyle Hidden `
      -RedirectStandardError $Log -RedirectStandardOutput "$Log.out"

    Write-Host -NoNewline '  waiting for grinder to be ready'
    for ($i = 0; $i -lt 45; $i++) {
        try { if ((Invoke-WebRequest 'http://127.0.0.1:8081/health' -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200) { break } } catch {}
        Write-Host -NoNewline '.'
        Start-Sleep -Seconds 2
    }
    Write-Host ''
    Write-Host '  Grinder ONLINE on :8081 (VRAM ~3.8 GB).' -ForegroundColor Green
    Write-Host ''
}

Start-Sleep -Seconds 3
