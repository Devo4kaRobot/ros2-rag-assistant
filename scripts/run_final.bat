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


echo === Tests ===
python -m pytest -q > results\logs\pytest.log 2>&1
type results\logs\pytest.log
echo.
echo === [1/3] Prompt v2 on dev questions (GigaChat-3-Ultra) ===
python src\rag\eval_generation.py --model GigaChat-3-Ultra --prompt v2 > results\logs\eval_dev_v2.log 2>&1
echo.
echo === [2/3] Held-out TEST questions: retrieval (BM25, e5, bge-m3, hybrid) ===
python src\rag\eval_retrieval.py configs\03_bm25_structural.yaml --split test > results\logs\test_retrieval.log 2>&1
python src\rag\eval_retrieval.py configs\05_e5base_recursive.yaml --split test >> results\logs\test_retrieval.log 2>&1
python src\rag\eval_retrieval.py configs\10_bgem3_structural.yaml --split test >> results\logs\test_retrieval.log 2>&1
python src\rag\eval_retrieval.py configs\12_hybrid_bm25_bgem3_structural.yaml --split test >> results\logs\test_retrieval.log 2>&1
echo.
echo === [3/3] Held-out TEST questions: answers with prompt v1 and v2 ===
python src\rag\eval_generation.py --split test --model GigaChat-3-Ultra --prompt v1 > results\logs\eval_test_v1.log 2>&1
python src\rag\eval_generation.py --split test --model GigaChat-3-Ultra --prompt v2 > results\logs\eval_test_v2.log 2>&1
echo DONE %date% %time% > results\logs\final_done.txt
echo ============================================================
echo  Done. You can close this window and tell Claude "gotovo".
echo ============================================================
pause
