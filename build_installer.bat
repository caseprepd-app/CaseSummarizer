@echo off
REM ──────────────────────────────────────────────────────────────
REM  CasePrepd Build Script
REM  Builds the standalone app with PyInstaller, then creates
REM  the installer with Inno Setup.
REM
REM  Prerequisites:
REM    - Python virtual environment at .venv\
REM    - Inno Setup 6 installed (ISCC.exe on PATH or default location)
REM ──────────────────────────────────────────────────────────────

setlocal enabledelayedexpansion

echo ============================================
echo   CasePrepd Build Script
echo ============================================
echo.

REM ── Step 1: Activate virtual environment ─────────────────────
echo [1/4] Activating virtual environment...
if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found at .venv\
    echo Run: python -m venv .venv
    exit /b 1
)
call .venv\Scripts\activate.bat

REM ── Step 2: Ensure PyInstaller is installed ──────────────────
echo [2/4] Checking PyInstaller...
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo Installing PyInstaller...
    pip install pyinstaller
    if errorlevel 1 (
        echo ERROR: Failed to install PyInstaller
        exit /b 1
    )
)
echo PyInstaller OK.
echo.

REM ── Step 3: Run PyInstaller ──────────────────────────────────
echo [3/4] Building with PyInstaller...
echo This may take several minutes.
echo.
pyinstaller caseprepd.spec --noconfirm
if errorlevel 1 (
    echo.
    echo ERROR: PyInstaller build failed.
    exit /b 1
)

if not exist "dist\CasePrepd\CasePrepd.exe" (
    echo ERROR: Expected output dist\CasePrepd\CasePrepd.exe not found.
    exit /b 1
)
echo PyInstaller build complete: dist\CasePrepd\CasePrepd.exe
echo.

REM ── Step 4: Run Inno Setup compiler ─────────────────────────
echo [4/4] Building installer with Inno Setup...

REM Try common ISCC.exe locations
set "ISCC="
where iscc.exe >nul 2>&1 && set "ISCC=iscc.exe"

if defined ISCC goto :run_inno

if exist "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" (
    set "ISCC=C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
    goto :run_inno
)

if exist "C:\Program Files\Inno Setup 6\ISCC.exe" (
    set "ISCC=C:\Program Files\Inno Setup 6\ISCC.exe"
    goto :run_inno
)

echo.
echo WARNING: Inno Setup compiler ISCC.exe not found.
echo Install Inno Setup 6 from https://jrsoftware.org/isinfo.php
echo Then re-run this script, or compile manually:
echo   "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer\caseprepd.iss
echo.
echo PyInstaller build succeeded. You can still run dist\CasePrepd\CasePrepd.exe
exit /b 0

:run_inno
"%ISCC%" installer\caseprepd.iss
if errorlevel 1 (
    echo.
    echo ERROR: Inno Setup compilation failed.
    exit /b 1
)

echo.
echo ============================================
echo   Build complete!
echo   Installer: installer\Output\CasePrepdSetup.exe
echo ============================================
