@echo off
REM Setup script for LLM Document Processing and QA System

echo Setting up LLM Document Processing and QA System...

REM Create directories
echo Creating directories...
mkdir data processed models 2>nul

REM Install Python dependencies
echo Installing Python dependencies...
pip install -r requirements.txt

echo Setup completed!
echo Please place your documents in the 'data' directory and run:
echo   python src/main.py --mode process
pause