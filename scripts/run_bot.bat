@echo off
rem Generated helper: runs project commands with the right environment and logs output.
cd /d "%~dp0.."
if not exist ".venv\Scripts\activate.bat" (
  echo [ERROR] .venv not found in %CD%
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
rem UTF-8 output so Cyrillic and symbols do not crash Python when redirected to a log file
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
rem Personal settings (e.g. a VPN proxy for Telegram/HuggingFace) live in scripts\local.bat, which is not in git.
rem GigaChat bypasses any proxy automatically (NO_PROXY in settings.py).
if exist "scripts\local.bat" call "scripts\local.bat"
if not exist "results\logs" mkdir "results\logs"

echo === Starting Telegram bot (stop: Ctrl+C or close this window) ===
python src\bot\main.py 2>&1
pause
