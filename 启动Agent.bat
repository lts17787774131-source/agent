@echo off
chcp 936 >nul
title 多工具智能助手 Agent 启动器
echo ==================================================
echo   多工具智能助手 Agent 一键启动
echo ==================================================
echo.
set "PY=C:\Python314\python.exe"
set "APP_DIR=D:\pythonProject5\多工具调用个人助手 Agent"
set "OLLAMA_APP=C:\Users\lts\AppData\Local\Programs\Ollama\ollama app.exe"

REM ===== 1. 检查并启动 Ollama =====
curl -s -o nul --noproxy "*" --max-time 3 http://127.0.0.1:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo [1/3] Ollama 未运行，正在启动...
    start "" "%OLLAMA_APP%"
    :wait_ollama
    ping -n 2 127.0.0.1 >nul
    curl -s -o nul --noproxy "*" --max-time 3 http://127.0.0.1:11434/api/tags >nul 2>&1
    if errorlevel 1 goto wait_ollama
    echo [1/3] Ollama 已就绪
) else (
    echo [1/3] Ollama 已在运行
)

REM ===== 2. 检查并启动 Agent 界面 =====
curl -s -o nul --noproxy "*" --max-time 3 http://127.0.0.1:7860 >nul 2>&1
if errorlevel 1 (
    echo [2/3] 启动 Agent 界面...
    start "Agent" /D "%APP_DIR%" "%PY%" app.py
    echo [2/3] 界面启动中，等待就绪...
    ping -n 8 127.0.0.1 >nul
) else (
    echo [2/3] Agent 界面已在运行
)

REM ===== 3. 打开浏览器 =====
echo [3/3] 打开界面 http://127.0.0.1:7860
start "" "http://127.0.0.1:7860"

echo.
echo 完成！如需停止，关闭 Agent 窗口即可。
echo.
pause