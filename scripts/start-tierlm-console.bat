@echo off
rem ==================================================================================================================
rem  tierlm from the console: loads one model with the settings below and serves it, with no browser on this PC.
rem  Edit the settings (right-click this file, Edit), save, then double-click it. Close the window to stop tierlm.
rem  Examples for common setups are under the settings. Every option: docs\CONSOLE.md, or tierlm-server.exe --help
rem ==================================================================================================================
setlocal
cd /d "%~dp0"

rem ---------------------------------------------- SETTINGS --------------------------------------------------------
rem  MODEL    a .gguf file (the first file, for a model in several), or its name in the models folder as the page
rem           lists it, e.g. unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf
rem  SAVED    yes: start from the settings the page saved for this model (what Find the fastest settings found);
rem           the settings below still win where they're filled in
rem  CTX      context length in tokens, per connection (32768 = 32K). Apps like Open WebUI need 32K or more for their
rem           own instructions and tools. Empty with SAVED=yes: the saved one
rem  OPTIONS  anything else, e.g. --experts follow --guesses 2 --threads 6 --kv 8 (see the examples below)
rem  HOST     empty: as set on the page (Allow other devices); 127.0.0.1: this PC only; 0.0.0.0: other devices too
rem  PORT     the port apps connect to
rem  PAGE     yes: whoever opens this PC's address in a browser gets tierlm's chat page (it costs this PC nothing
rem           until someone opens it); no: the API only. No browser opens on this PC either way
rem  KEY      an API key other devices must send; empty: none. Set one when HOST is 0.0.0.0, unless the network is
rem           yours alone

set "MODEL=C:\path\to\your-model.gguf"
set "SAVED=no"
set "CTX=32768"
set "OPTIONS="
set "HOST="
set "PORT=8040"
set "PAGE=yes"
set "KEY="

rem ---------------------------------------------- EXAMPLES --------------------------------------------------------
rem  Copy the lines of the one you want over the settings above (MODEL stays yours).
rem
rem  1. Just this PC, with your own app on it (Open WebUI Desktop, opencode, a script on this PC):
rem       set "HOST=127.0.0.1"   set "PAGE=no"    set "KEY="
rem     The app's base URL: http://127.0.0.1:8040/v1
rem
rem  2. This PC as the AI server; other PCs chat in their own browser:
rem       set "HOST=0.0.0.0"     set "PAGE=yes"   set "KEY=choose-a-long-key"
rem     tierlm prints the address to open on the other PCs, e.g. http://192.168.1.50:8040 (the page asks for the key).
rem
rem  3. This PC as the AI server; other PCs use their own apps (Open WebUI, opencode, SillyTavern):
rem       set "HOST=0.0.0.0"     set "PAGE=no"    set "KEY=choose-a-long-key"
rem     Their base URL: http://<the address tierlm prints>/v1, with the key.
rem
rem  4. Both, for several people at once (each connection gets its own CTX of context memory):
rem       set "HOST=0.0.0.0"     set "PAGE=yes"   set "KEY=choose-a-long-key"   set "OPTIONS=--connections 2"
rem
rem  Speed, once the model is chosen:
rem    - Tuned on the page first (Find the fastest settings):   set "SAVED=yes"   set "CTX="
rem    - A MoE model bigger than the card's VRAM, 8 GB card, 6-core laptop:
rem         set "OPTIONS=--experts follow --guesses 2 --threads 6"
rem    - The same on a 32 GB card with a 16-core CPU, its draft (mtp-) file next to the model:
rem         set "OPTIONS=--experts follow --guesses 4 --threads 12"
rem    - A longer context in less memory:   set "CTX=65536"   set "OPTIONS=--kv 8"
rem ------------------------------------------------------------------------------------------------------------------

if /i "%MODEL%"=="C:\path\to\your-model.gguf" (
  echo.
  echo  Edit start-tierlm-console.bat first: put your model on the "set MODEL=" line, save, and run it again.
  echo  ^(Right-click the file, Edit. Every option is in docs\CONSOLE.md.^)
  echo.
  pause
  exit /b 1
)
set "ARGS=--port %PORT%"
if /i not "%PAGE%"=="yes" set "ARGS=%ARGS% --no-web"
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
