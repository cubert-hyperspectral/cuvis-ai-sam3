@echo off
setlocal enabledelayedexpansion

:: ============================================================================
:: Build script for CuvisAI SAM3 Server installer
:: ============================================================================

cd /d "%~dp0\.."
set "PROJECT_ROOT=%cd%"
set "INSTALLER_DIR=%PROJECT_ROOT%\installer"

echo ============================================================
echo  CuvisAI SAM3 Server - Windows Installer Build
echo ============================================================
echo.
echo Project root: %PROJECT_ROOT%
echo.

echo [1/8] Installing build dependencies...
uv pip install pyinstaller>=6.0.0 pystray>=0.19 Pillow>=10.0.0 tqdm>=4.0 huggingface_hub>=0.36.0
if errorlevel 1 (
    echo ERROR: Failed to install build dependencies.
    exit /b 1
)
echo.

echo [2/8] Converting icon to ICO...
uv run python "%INSTALLER_DIR%\convert_icon.py"
if errorlevel 1 (
    echo ERROR: Icon conversion failed.
    exit /b 1
)
echo.

echo [3/8] Detecting package version...
for /f "delims=" %%v in ('uv run python -c "from importlib.metadata import version; print(version(\"cuvis-ai-sam3\"))"') do set "APP_VERSION=%%v"
if "%APP_VERSION%"=="" (
    echo WARNING: Could not detect version, using 0.0.0
    set "APP_VERSION=0.0.0"
)
echo       Version: %APP_VERSION%
echo.

echo [4/8] Building server bundle (PyInstaller)...
echo       This may take a while (torch + triton + timm)...
uv run pyinstaller --noconfirm --distpath "%PROJECT_ROOT%\dist" --workpath "%PROJECT_ROOT%\build" "%INSTALLER_DIR%\sam3_server.spec"
if errorlevel 1 (
    echo ERROR: sam3_server.spec build failed.
    exit /b 1
)
echo.

echo [5/8] Building tray bundle (PyInstaller)...
uv run pyinstaller --noconfirm --distpath "%PROJECT_ROOT%\dist" --workpath "%PROJECT_ROOT%\build" "%INSTALLER_DIR%\sam3_tray.spec"
if errorlevel 1 (
    echo ERROR: sam3_tray.spec build failed.
    exit /b 1
)
echo.

echo [6/8] Building download-weights bundle (PyInstaller)...
uv run pyinstaller --noconfirm --distpath "%PROJECT_ROOT%\dist" --workpath "%PROJECT_ROOT%\build" "%INSTALLER_DIR%\download_weights.spec"
if errorlevel 1 (
    echo ERROR: download_weights.spec build failed.
    exit /b 1
)
echo.

echo [7/8] Verifying build outputs...
if not exist "%PROJECT_ROOT%\dist\sam3-server\sam3-rest-api.exe" (
    echo ERROR: Missing dist\sam3-server\sam3-rest-api.exe
    exit /b 1
)
if not exist "%PROJECT_ROOT%\dist\sam3-tray\sam3-tray.exe" (
    echo ERROR: Missing dist\sam3-tray\sam3-tray.exe
    exit /b 1
)
if not exist "%PROJECT_ROOT%\dist\download-weights\download-weights.exe" (
    echo ERROR: Missing dist\download-weights\download-weights.exe
    exit /b 1
)
echo       Dist outputs verified.
echo.

echo [8/8] Building installer with Inno Setup...
set "ISCC="
if exist "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" (
    set "ISCC=C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
)
if exist "C:\Program Files\Inno Setup 6\ISCC.exe" (
    set "ISCC=C:\Program Files\Inno Setup 6\ISCC.exe"
)

if "%ISCC%"=="" (
    echo WARNING: Inno Setup 6 not found. Skipping installer creation.
    echo          Dist bundles are available under dist\.
    goto :done
)

"%ISCC%" /DMyAppVersion=%APP_VERSION% "%INSTALLER_DIR%\sam3_server.iss"
if errorlevel 1 (
    echo ERROR: Inno Setup compilation failed.
    exit /b 1
)
echo.

:done
echo ============================================================
echo  Build complete!
echo ============================================================
if exist "%INSTALLER_DIR%\Output\CuvisAI-SAM3-Server-%APP_VERSION%-Setup.exe" (
    echo  Installer: installer\Output\CuvisAI-SAM3-Server-%APP_VERSION%-Setup.exe
) else (
    echo  Dist bundles ready at dist\sam3-server, dist\sam3-tray, dist\download-weights
)
echo.
endlocal

