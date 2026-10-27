@echo off
rem tierlm launcher: starts the server with the first model found in the "models" folder
rem and opens the web page. Close this window (or press Ctrl+C) to stop tierlm.
setlocal
cd /d "%~dp0"
set "MODEL="
for %%f in ("models\*.gguf") do if not defined MODEL set "MODEL=%%~ff"
if not defined MODEL (
  echo.
  echo  No model found. Put a supported .gguf file in the "models" folder next to this file, for example:
  echo    OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf
  echo  Download it from:
  echo    https://huggingface.co/bartowski/OLMoE-1B-7B-0924-Instruct-GGUF
  echo.
  pause
  exit /b 1
)
set "PORT=8040"
if not "%~1"=="" set "PORT=%~1"
echo.
echo  Starting tierlm with: %MODEL%
echo  The web page will open at http://127.0.0.1:%PORT% in a few seconds.
echo.
start "" /b cmd /c "ping -n 9 127.0.0.1 >nul & start http://127.0.0.1:%PORT%"
tierlm-server.exe "%MODEL%" --port %PORT% --web web
echo.
echo  tierlm has stopped.
pause
