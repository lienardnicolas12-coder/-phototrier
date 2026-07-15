@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
python tri_photos_automatique_gui.py
pause
