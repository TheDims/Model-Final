@echo off
REM SolarHome Forecast starten zonder VS Code: dubbelklik op dit bestand.
REM De app opent vanzelf in je browser (http://localhost:8501). Sluit dit venster om te stoppen.
title SolarHome Forecast
cd /d "%~dp0"
set PY=%USERPROFILE%\anaconda3\python.exe
if not exist "%PY%" set PY=python
"%PY%" -m streamlit run app\app.py
pause
