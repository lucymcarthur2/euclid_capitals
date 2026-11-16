@echo off
rem Initials Studio - double-click to start. First start installs everything (needs internet, ~5-10 min).
setlocal
cd /d "%~dp0"
title Initials Studio

if exist ".venv\Scripts\python.exe" goto run

echo.
echo  Setting up Initials Studio for the first time...
echo  (this downloads about 300 MB and takes a few minutes; later starts are instant)
echo.
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY (where python >nul 2>nul && set "PY=python")
if not defined PY (
  echo  Python 3.10 or newer is needed. Install it from https://www.python.org/downloads/
  echo  ^(tick "Add python.exe to PATH" during installation^), then double-click this file again.
  pause
  exit /b 1
)
%PY% -m venv .venv || (echo Could not create the Python environment. & pause & exit /b 1)
".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
".venv\Scripts\python.exe" -m pip install -r requirements.txt || (echo Installation failed - see the messages above. & pause & exit /b 1)

:run
".venv\Scripts\python.exe" -m initials_studio %*
if errorlevel 1 pause
