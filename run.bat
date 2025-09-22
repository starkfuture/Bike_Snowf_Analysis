@echo off
echo Checking Python installation...

if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv venv
)

call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo Failed to activate virtual environment.
    pause
    exit /b
)

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing dependencies...
pip install -r requirements.txt

echo Launching the Dash app...

start "" "http://127.0.0.1:8050"  :: This opens the default browser at the app URL
python Analyzer_2.py

pause



