@echo off
rem tierlm launcher: starts the server and opens the web page, where you choose
rem your hardware and model. Close this window (or press Ctrl+C) to stop tierlm.
rem Optional: start-tierlm.bat <port>   (default 8040)
setlocal
cd /d "%~dp0"
set "PORT=8040"
if not "%~1"=="" set "PORT=%~1"
echo.
echo  Starting tierlm...
echo  Your browser will open http://127.0.0.1:%PORT% in a few seconds.
echo  Choose your hardware and model there. Keep this window open while you use tierlm.
echo.
start "" /b cmd /c "ping -n 4 127.0.0.1 >nul & start http://127.0.0.1:%PORT%"
tierlm-server.exe --models models --web web --port %PORT%
echo.
echo  tierlm has stopped. If it stopped straight away, the message above says why.
pause
