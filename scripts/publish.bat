@echo off
rem Publishes this project to GitHub. Before running: create an EMPTY repository named
rem ros2-rag-assistant on github.com (no README, no .gitignore, no license).
cd /d "%~dp0.."
if exist "scripts\local.bat" call "scripts\local.bat"

where git >nul 2>nul
if errorlevel 1 (
  echo Git is not installed. Installing Git for Windows via winget...
  winget install --id Git.Git -e --source winget --accept-package-agreements --accept-source-agreements
)
where git >nul 2>nul
if errorlevel 1 set "PATH=%PATH%;C:\Program Files\Git\cmd"
where git >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Git still not found. Install it from https://git-scm.com and run this file again.
  pause
  exit /b 1
)

echo.
set /p GHUSER=GitHub username: 
set /p GITNAME=Your name for commits and LICENSE (e.g. Ivan Ivanov): 
set /p GITEMAIL=Email for commits (Enter = private GitHub noreply address): 
if "%GITEMAIL%"=="" set "GITEMAIL=%GHUSER%@users.noreply.github.com"

rem CI workflow (kept in ci\ because remote tools cannot write into .github)
if not exist ".github\workflows" mkdir ".github\workflows"
copy /y "ci\github-actions-ci.yml" ".github\workflows\ci.yml" >nul

rem put the username into README badges/links and the name into LICENSE (UTF-8 without BOM)
powershell -NoProfile -Command "$u=New-Object System.Text.UTF8Encoding($false); $p=(Resolve-Path 'README.md').Path; $t=[IO.File]::ReadAllText($p,$u).Replace('OWNER','%GHUSER%'); [IO.File]::WriteAllText($p,$t,$u); $l=(Resolve-Path 'LICENSE').Path; $t=[IO.File]::ReadAllText($l,$u); if($t -notmatch 'Copyright \(c\) 2026 \S'){ $t=$t.Replace('Copyright (c) 2026','Copyright (c) 2026 %GITNAME%') }; [IO.File]::WriteAllText($l,$t,$u)"

if not exist ".git" git init -q -b main
git config user.name "%GITNAME%"
git config user.email "%GITEMAIL%"
git config core.autocrlf true
git add -A

rem safety: secrets must never be committed
git ls-files --cached | findstr /i /r "^\.env$ ^certs/ local\.bat$" >nul
if not errorlevel 1 (
  echo [ERROR] A secret file is staged ^(.env / certs / local.bat^). Aborting.
  git ls-files --cached | findstr /i /r "^\.env$ ^certs/ local\.bat$"
  pause
  exit /b 1
)

git commit -q -m "ROS 2 RAG assistant: corpus, eval set, retrieval ablations, GigaChat answers, Telegram bot"
git remote remove origin >nul 2>nul
git remote add origin https://github.com/%GHUSER%/ros2-rag-assistant.git
if not exist "results\logs" mkdir "results\logs"
git ls-files > results\logs\published_files.txt
echo.
echo Pushing to https://github.com/%GHUSER%/ros2-rag-assistant ...
echo (a browser window may open to sign in to GitHub - that is expected)
git push -u origin main
if errorlevel 1 (
  echo [ERROR] Push failed. Check that the empty repository exists and you are signed in.
  echo push failed > results\logs\publish_status.txt
) else (
  echo pushed https://github.com/%GHUSER%/ros2-rag-assistant > results\logs\publish_status.txt
  echo ============================================================
  echo  Done! Open https://github.com/%GHUSER%/ros2-rag-assistant
  echo ============================================================
)
pause
