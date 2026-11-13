@echo off
rem tierlm without the web page or a browser: loads one model with the settings below and serves the API for your
rem own front end (Open WebUI, SillyTavern, a coding agent, a script). Edit the lines below, save, then double-click.
rem Every option is in docs\CONSOLE.md (or run: tierlm-server.exe --help). Close this window to stop tierlm.
setlocal
cd /d "%~dp0"

rem ---- The model: a .gguf file (the first file, for a model in several), or its name in the models folder
rem      as the page lists it, e.g. unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf
set "MODEL=C:\path\to\your-model.gguf"

rem ---- Start from the settings the page saved for this model (what Find the fastest settings found): yes or no.
rem      The lines below still win where they're filled in; leave CTX empty to keep the saved context too.
set "SAVED=no"

rem ---- Context length in tokens, per connection (32768 = 32K). Apps like Open WebUI need room for their own
rem      instructions and tools: 32K or more.
set "CTX=32768"

rem ---- Any other options, e.g. --experts follow --guesses 2 --threads 6 --kv 8 (see docs\CONSOLE.md)
set "OPTIONS="

rem ---- Where to listen: empty (as set on the page: Allow other devices), 127.0.0.1 (this computer only) or 0.0.0.0
rem      (other devices on your network too); and the port
set "HOST="
set "PORT=8040"

rem ---- A key other devices must send (empty: none). Set one when HOST is 0.0.0.0, unless the network is yours alone.
set "KEY="

rem ------------------------------------------------------------------------------------------------------------------
if /i "%MODEL%"=="C:\path\to\your-model.gguf" (
  echo.
  echo  Edit start-tierlm-console.bat first: put your model on the "set MODEL=" line, save, and run it again.
  echo  ^(Right-click the file, Edit. Every option is in docs\CONSOLE.md.^)
  echo.
  pause
  exit /b 1
)
set "ARGS=--no-web --port %PORT%"
if not "%HOST%"=="" set "ARGS=%ARGS% --host %HOST%"
if /i "%SAVED%"=="yes" set "ARGS=%ARGS% --saved"
if not "%CTX%"=="" set "ARGS=%ARGS% --ctx %CTX%"
if not "%KEY%"=="" set "ARGS=%ARGS% --api-key %KEY%"
echo.
echo  Starting tierlm with %MODEL%
echo  When it says it's ready, point your app at http://127.0.0.1:%PORT%/v1
echo  ^(from another device: this PC's network address, which tierlm prints when other devices may connect^)
echo.
tierlm-server.exe "%MODEL%" %ARGS% %OPTIONS%
echo.
echo  tierlm has stopped. If it stopped straight away, the message above says why.
pause
