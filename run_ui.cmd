@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_ui.ps1" %*
exit /b %ERRORLEVEL%
