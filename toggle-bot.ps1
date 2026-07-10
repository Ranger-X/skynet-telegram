# =====================================================================
#  T-800 bot toggle — one click to start/stop the whole LOCAL stack.
#
#  Running  -> stops the bot AND llama-server (frees ~7GB RAM on the laptop).
#  Stopped  -> starts llama-server (waits until it's ready) then the bot.
#
#  state.json is NEVER touched, so participants / quotes / recent-message
#  history survive across toggles. (Only the live 16-message chat context
#  resets on restart — that's ephemeral by design.)
#
#  One engine now: llama-server does text + vision + audio (mmproj bf16 projector loaded in). Ollama is
#  no longer used by the bot.
#
#  Double-click `toggle-bot.cmd` to run this, or right-click -> Run with PowerShell.
# =====================================================================

$ErrorActionPreference = 'SilentlyContinue'

# --- paths ---------------------------------------------------------------
$Proj      = 'c:\Users\Impi\Desktop\Coding\Projects\skynet-telegram'
$Py        = Join-Path $Proj '.venv\Scripts\python.exe'
$BotLog    = Join-Path $Proj 'bot.log'
$LlamaExe  = 'D:\llamacpp\llama-server.exe'
$LlamaDir  = 'D:\llamacpp'
$LlamaLog  = 'D:\llamacpp\server.log'
# Official bf16 projector (our own F16 export is broken). --no-warmup required or gemma4's warmup graph
# crashes with the multimodal context. This one server serves text + vision + audio.
$Mmproj    = 'D:\ollama-models\blobs\sha256-675ad6e68101ca9413ec806855c452362f0213f2dfc5800996b086fdb8119842'
# KV cache quantized to q8_0 (both K and V; V-quant needs flash-attention on) — frees ~1-1.5 GB
# of RAM vs fp16 KV at 8192 ctx, which is what lets the RAG embedder/reranker breathe next to
# the 12B. Quality impact at q8_0 is negligible (verified on our prompts 2026-07-09).
$LlamaArgs = @(
    # heretic = decensored gemma-4-12b finetune (same arch, official mmproj still applies);
    # switched from the stock model 2026-07-09 — it moralized at the chat's humor.
    '-m','C:\Users\Impi\.lmstudio\models\igorls\gemma-4-12B-it-heretic-GGUF\gemma-4-12B-it-heretic-Q4_K_M.gguf',
    '--mmproj',$Mmproj,'--no-warmup',
    '-ngl','0','-c','8192','-np','2','-t','6','--jinja',
    '-fa','on','-ctk','q8_0','-ctv','q8_0',
    '--host','127.0.0.1','--port','8080'
)

function Get-Bot   { Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*main.py*' } }
function Get-Llama { Get-CimInstance Win32_Process -Filter "Name='llama-server.exe'" }

# --- toggle --------------------------------------------------------------
if (Get-Bot) {
    # ---------- STOP ----------
    Write-Host ''
    Write-Host '  Stopping T-800...' -ForegroundColor Yellow
    Get-Bot   | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
    Get-Llama | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
    Write-Host '  T-800 OFFLINE. RAM freed. (history in state.json preserved)' -ForegroundColor Green
    Write-Host ''
}
else {
    # ---------- START ----------
    Write-Host ''
    Write-Host '  Starting T-800...' -ForegroundColor Yellow
    $env:PYTHONIOENCODING = 'utf-8'   # inherited by the child process -> clean UTF-8 logs

    if (-not (Get-Llama)) {
        Write-Host '  launching llama-server (text + vision + audio)...'
        # stderr -> the main log (llama.cpp logs to stderr); stdout -> a near-empty sidecar.
        Start-Process -FilePath $LlamaExe -ArgumentList $LlamaArgs -WorkingDirectory $LlamaDir `
            -WindowStyle Hidden -RedirectStandardError $LlamaLog -RedirectStandardOutput "$LlamaLog.out"

        Write-Host -NoNewline '  waiting for llama-server to be ready'
        $ready = $false
        for ($i = 0; $i -lt 90; $i++) {
            try { if ((Invoke-WebRequest 'http://127.0.0.1:8080/health' -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200) { $ready = $true; break } } catch {}
            Write-Host -NoNewline '.'
            Start-Sleep -Seconds 2
        }
        Write-Host ''
        if (-not $ready) { Write-Host '  WARNING: llama-server not responding — bot will fall back to cloud for text.' -ForegroundColor Red }
    }
    else {
        Write-Host '  llama-server already running.'
    }

    # stderr -> bot.log (Python's logging writes to stderr); stdout -> a near-empty sidecar.
    Start-Process -FilePath $Py -ArgumentList 'main.py' -WorkingDirectory $Proj `
        -WindowStyle Hidden -RedirectStandardError $BotLog -RedirectStandardOutput "$BotLog.out"
    Write-Host '  T-800 ONLINE. (logs -> bot.log)' -ForegroundColor Green
    Write-Host ''
}

Start-Sleep -Seconds 3   # give a double-click user time to read the status before the window closes
