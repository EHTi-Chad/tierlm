@echo off
rem Builds a portable Windows release of tierlm and zips it into dist\.
rem Run from a Visual Studio Developer Command Prompt, in the project folder, with the CUDA toolkit (12.8 or newer,
rem for RTX 50-series) and the Vulkan SDK installed:
rem     scripts\package-windows.bat 0.2.0
rem One download for every PC: it contains both GPU engines and picks one at start-up. NVIDIA GeForce RTX cards
rem use CUDA (fastest on NVIDIA); Intel Arc, AMD Radeon and older NVIDIA cards use Vulkan; with no GPU, the CPU.
rem NVIDIA's CUDA runtime DLLs are bundled, so users install nothing but their normal graphics driver.
rem     scripts\package-windows.bat 0.2.0 lite      tierlm-0.2.0-windows-lite.zip: Vulkan only, about 55 MB. Runs on
rem                                                Intel, AMD and NVIDIA GPUs (slower than CUDA on NVIDIA). Needs no CUDA toolkit to build.
rem     scripts\package-windows.bat 0.2.0 both      both of the above
setlocal
if "%~1"=="" ( echo usage: scripts\package-windows.bat VERSION [lite^|both]   e.g. 0.2.0 both & exit /b 1 )
set "VER=%~1"
set "FLAVOR=%~2"
if "%FLAVOR%"=="" set "FLAVOR=full"
if /i "%FLAVOR%"=="vulkan" set "FLAVOR=lite"
cd /d "%~dp0.."
if /i "%FLAVOR%"=="both" (
  call "%~f0" %VER% full || exit /b 1
  call "%~f0" %VER% lite || exit /b 1
  exit /b 0
)

if /i "%FLAVOR%"=="full" (
  set "NAME=tierlm-%VER%-windows"
  set "BDIR=build-release"
  echo == Building ^(portable: AVX2 baseline, static runtime, CUDA for RTX 20/30/40/50 + Vulkan^) ==
  rem CUDA architectures: 75 = RTX 20, 86 = RTX 30, 89 = RTX 40, 120 = RTX 50
  cmake -S . -B build-release -G Ninja -DCMAKE_BUILD_TYPE=Release -DTIERLM_CUDA=ON -DTIERLM_VULKAN=ON -DTIERLM_PORTABLE=ON "-DCMAKE_CUDA_ARCHITECTURES=75;86;89;120" -DTIERLM_VERSION=%VER% || exit /b 1
) else if /i "%FLAVOR%"=="lite" (
  set "NAME=tierlm-%VER%-windows-lite"
  set "BDIR=build-release-vulkan"
  echo == Building ^(portable: AVX2 baseline, static runtime, Vulkan only^) ==
  cmake -S . -B build-release-vulkan -G Ninja -DCMAKE_BUILD_TYPE=Release -DTIERLM_VULKAN=ON -DTIERLM_PORTABLE=ON -DTIERLM_VERSION=%VER% || exit /b 1
) else ( echo unknown option "%FLAVOR%": leave it out, or use lite or both & exit /b 1 )
cmake --build %BDIR% || exit /b 1

echo == Collecting files ==
set "OUT=dist\%NAME%"
if exist "%OUT%" rmdir /s /q "%OUT%"
mkdir "%OUT%\web" "%OUT%\models" "%OUT%\licenses" || exit /b 1
for %%t in (server run bench devices inspect plan kv) do copy /y "%BDIR%\tierlm-%%t.exe" "%OUT%\" >nul || exit /b 1
copy /y web\index.html "%OUT%\web\" >nul
if /i "%FLAVOR%"=="full" (
  rem NVIDIA's redistributable runtime: cudart, cuBLAS, cuBLASLt (CUDA 12 keeps them in bin, CUDA 13 in bin\x64)
  for %%d in ("%CUDA_PATH%\bin" "%CUDA_PATH%\bin\x64") do (
    for %%f in ("%%~d\cudart64_*.dll" "%%~d\cublas64_*.dll" "%%~d\cublasLt64_*.dll") do copy /y "%%~f" "%OUT%\" >nul
  )
  if not exist "%OUT%\cublasLt64_*.dll" ( echo Could not find the CUDA runtime DLLs under "%CUDA_PATH%". & exit /b 1 )
  if exist "%CUDA_PATH%\EULA.txt" copy /y "%CUDA_PATH%\EULA.txt" "%OUT%\licenses\NVIDIA-CUDA-EULA.txt" >nul
)
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
