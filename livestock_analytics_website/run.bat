@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.10 or newer and enable Add Python to PATH.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  python -m venv .venv
  if errorlevel 1 goto failed
)

echo Installing or checking project packages...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed

echo.
echo Starting LivestockLens...
echo Open http://127.0.0.1:5000 in your browser.
echo Keep this window open while using the website.
echo.
".venv\Scripts\python.exe" app.py
pause
exit /b 0

:failed
echo.
echo Setup failed. Check your internet connection and the error message above.
pause
exit /b 1
