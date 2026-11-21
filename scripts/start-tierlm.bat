@echo off
rem tierlm launcher: starts the server and opens the web page, where you choose
rem your model. tierlm picks the hardware itself. Close this window (or press Ctrl+C) to stop tierlm.
rem Optional: start-tierlm.bat <port>   (default 8040)
rem Tracking down a problem: start-tierlm-with-log.bat starts tierlm the same way with both of these set to yes.
rem   DETAILED  yes: more detail in this window (the engines' own lines, each request to tierlm, this PC)
rem   SAVE_LOG  yes: also save this window's text to a new file in the logs folder, to send in with a problem report
setlocal
cd /d "%~dp0"
if not defined DETAILED set "DETAILED=no"
if not defined SAVE_LOG set "SAVE_LOG=no"
set "PORT=8040"
if not "%~1"=="" set "PORT=%~1"
set "LOGARGS="
if /i "%DETAILED%"=="yes" set "LOGARGS=%LOGARGS% --detailed-log"
if /i "%SAVE_LOG%"=="yes" set "LOGARGS=%LOGARGS% --log-file auto"
echo.
echo  Starting tierlm...
echo  Your browser will open http://127.0.0.1:%PORT% in a few seconds.
echo  Choose your model there. Keep this window open while you use tierlm.
echo.
start "" /b cmd /c "ping -n 4 127.0.0.1 >nul & start http://127.0.0.1:%PORT%"
rem The models folder defaults to "models" here; choosing another one on the page is remembered
rem in tierlm-settings.json.
tierlm-server.exe --web web --port %PORT%%LOGARGS%
echo.
echo  tierlm has stopped. If it stopped straight away, the message above says why.
if /i "%SAVE_LOG%"=="yes" (
  echo  This window's text is saved in the logs folder ^(the newest file^), which opens now.
  start "" explorer "%~dp0logs"
)
pause
