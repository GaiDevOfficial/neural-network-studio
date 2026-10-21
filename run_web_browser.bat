@echo off
title Node Studio Web Server
echo ========================================================
echo   Starting Node Studio...
echo ========================================================
echo Opening http://127.0.0.1:8080/graph_studio.html in default browser...
start http://127.0.0.1:8080/graph_studio.html
python server.py
pause
