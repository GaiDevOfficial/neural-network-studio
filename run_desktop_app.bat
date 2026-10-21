@echo off
title Neural Network Studio Launcher
echo ========================================================
echo   Launching Neural Network Studio Desktop Application...
echo ========================================================
set "APP=dist\NeuralNetworkStudio\NeuralNetworkStudio.exe"
if exist "%APP%" (
    start "" "%APP%"
    goto :eof
)
echo [ERROR] The portable app was not found in dist\NeuralNetworkStudio\.
echo Running the build pipeline now...
python desktop\build_exe.py
if exist "%APP%" (
    start "" "%APP%"
) else (
    echo [ERROR] Build did not produce %APP%.
    pause
)
