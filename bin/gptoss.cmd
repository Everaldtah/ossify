@echo off
powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" gptoss %*
