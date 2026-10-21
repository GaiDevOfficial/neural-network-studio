@echo off
title Rebuilding Neural Network Studio Desktop App
echo ========================================================
echo   Rebuilding Portable Windows App and Web Bundle
echo ========================================================
python desktop/build_exe.py
echo.
echo Process complete. Press any key to exit.
pause >nul
