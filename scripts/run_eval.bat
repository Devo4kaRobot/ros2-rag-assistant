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

echo === Step 3: tests ===
python -m pytest -q > results\logs\pytest.log 2>&1
type results\logs\pytest.log
echo.
echo === Step 3: evaluating answers with GigaChat-3-Ultra (about 5 min) ===
python src\rag\eval_generation.py --model GigaChat-3-Ultra > results\logs\eval_GigaChat-3-Ultra.log 2>&1
type results\logs\eval_GigaChat-3-Ultra.log | findstr /C:"correct_refusal" /C:"context_hit" /C:"Error" /C:"error"
echo.
echo === Step 3: evaluating answers with GigaChat-2 (about 5 min) ===
python src\rag\eval_generation.py --model GigaChat-2 > results\logs\eval_GigaChat-2.log 2>&1
type results\logs\eval_GigaChat-2.log | findstr /C:"correct_refusal" /C:"context_hit" /C:"Error" /C:"error"
echo.
echo DONE %date% %time% > results\logs\eval_done.txt
echo ============================================================
echo  Done. You can close this window and tell Claude "gotovo".
echo ============================================================
pause
