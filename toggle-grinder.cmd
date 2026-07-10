@echo off
REM Start/stop the GPU vision grinder (Qwen3-VL-4B on :8081). Runs toggle-grinder.ps1.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0toggle-grinder.ps1"
