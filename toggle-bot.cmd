@echo off
REM Double-click this to start/stop the T-800 local stack. It just runs toggle-bot.ps1
REM next to it, bypassing the default "open .ps1 in editor" behaviour.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0toggle-bot.ps1"
