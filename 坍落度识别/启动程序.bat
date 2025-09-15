@echo off
chcp 65001 >nul
title 坍落度识别系统

:: 设置环境变量解决OpenMP冲突
set KMP_DUPLICATE_LIB_OK=TRUE

echo ========================================
echo    坍落度识别系统 - 启动程序
echo ========================================
echo ✅ 已设置 KMP_DUPLICATE_LIB_OK=TRUE 解决OpenMP冲突
echo.

:: 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 错误: 未找到Python，请先安装Python 3.7或更高版本
    echo    下载地址: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

:: 显示Python版本
echo ✅ 检测到Python:
python --version

:: 检查是否已安装依赖
echo.
echo 🔍 检查依赖包...
python -c "import torch, cv2, PIL, matplotlib" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  检测到缺少依赖包，正在自动安装...
    echo.
    python install.py
    if errorlevel 1 (
        echo ❌ 依赖安装失败，请手动运行 install.py
        pause
        exit /b 1
    )
) else (
    echo ✅ 依赖包检查通过
)

:: 启动程序
echo.
echo 🚀 启动坍落度识别系统...
echo.
python slump_recognition_gui.py

:: 程序退出处理
if errorlevel 1 (
    echo.
    echo ❌ 程序运行出错
    echo 请检查日志文件: logs\slump_recognition_*.log
) else (
    echo.
    echo ✅ 程序正常退出
)

echo.
pause
