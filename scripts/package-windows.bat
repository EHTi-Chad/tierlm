@echo off
rem Builds a portable Windows release of tierlm and zips it into dist\.
rem Run from a Visual Studio Developer Command Prompt, in the project folder, with the Vulkan SDK installed:
rem     scripts\package-windows.bat 0.1.0
setlocal
if "%~1"=="" ( echo usage: scripts\package-windows.bat VERSION   e.g. 0.1.0 & exit /b 1 )
set "VER=%~1"
set "NAME=tierlm-%VER%-windows-x64-vulkan"
cd /d "%~dp0.."

echo == Building (portable: AVX2 baseline, static runtime, Vulkan) ==
cmake -S . -B build-release -G Ninja -DCMAKE_BUILD_TYPE=Release -DTIERLM_VULKAN=ON -DTIERLM_PORTABLE=ON -DTIERLM_VERSION=%VER% || exit /b 1
cmake --build build-release || exit /b 1

echo == Collecting files ==
set "OUT=dist\%NAME%"
if exist "%OUT%" rmdir /s /q "%OUT%"
mkdir "%OUT%\web" "%OUT%\models" "%OUT%\licenses" || exit /b 1
for %%t in (server run bench devices inspect plan kv) do copy /y "build-release\tierlm-%%t.exe" "%OUT%\" >nul || exit /b 1
copy /y web\index.html "%OUT%\web\" >nul
copy /y scripts\start-tierlm.bat "%OUT%\" >nul
mkdir "%OUT%\docs" && copy /y docs\*.md "%OUT%\docs\" >nul
(echo tierlm %VER%& echo.& echo Start here: docs\GETTING_STARTED.md& echo Problems: docs\FAQ.md& echo Project page: https://github.com/EHTi-Chad/tierlm& echo.& echo 1. Put a supported .gguf model in the models folder.& echo 2. Double-click start-tierlm.bat.) > "%OUT%\README-FIRST.txt"
copy /y LICENSE "%OUT%\licenses\tierlm-LICENSE.txt" >nul
copy /y THIRD_PARTY_NOTICES.md "%OUT%\licenses\" >nul
copy /y third_party\ggml\LICENSE "%OUT%\licenses\ggml-LICENSE.txt" >nul
copy /y third_party\httplib\LICENSE "%OUT%\licenses\cpp-httplib-LICENSE.txt" >nul
copy /y third_party\json\LICENSE.MIT "%OUT%\licenses\nlohmann-json-LICENSE.txt" >nul
echo Put your .gguf model file in this folder.> "%OUT%\models\PUT-MODEL-HERE.txt"

echo == Zipping ==
if exist "dist\%NAME%.zip" del "dist\%NAME%.zip"
powershell -NoProfile -Command "Compress-Archive -Path 'dist\%NAME%\*' -DestinationPath 'dist\%NAME%.zip'" || exit /b 1
powershell -NoProfile -Command "(Get-FileHash 'dist\%NAME%.zip' -Algorithm SHA256).Hash.ToLower() + '  %NAME%.zip' | Out-File -Encoding ascii 'dist\%NAME%.zip.sha256'"
echo.
echo Done: dist\%NAME%.zip
type "dist\%NAME%.zip.sha256"
