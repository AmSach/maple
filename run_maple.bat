@echo off
chcp 65001 >nul
title MAPLE: Gemma 4 26B-A4B Engine (RTX 4050 6GB)

echo ===============================================================================
echo   MAPLE: Manifold-Activated Pruning of Latent Experts
echo   Model: Gemma 4 26B-A4B (48 Experts, 4.75 GB, 1.48 BPW)
echo   GPU:   100%% In-VRAM GPU Execution (NVIDIA RTX 4050 6GB)
echo ===============================================================================
echo.

set "PATH=C:\Users\amans\AppData\Local\Programs\Ollama\lib\ollama;D:\tools\llama-cpp;%PATH%"

set "PY_BIN=E:\Python312\python.exe"
if not exist "%PY_BIN%" (
    set "PY_BIN=python"
)

cd /d "%~dp0engine"
"%PY_BIN%" "chat.py"

if %ERRORLEVEL% neq 0 (
    echo.
    echo [!] Process exited with code %ERRORLEVEL%.
    pause
)
