@echo off
REM Resume/run the NAS100 Dukascopy downloader (resumes automatically).
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" "scripts\run.py" %*
