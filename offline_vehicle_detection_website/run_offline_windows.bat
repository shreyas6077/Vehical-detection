@echo off
title Offline Vehicle Detection Studio (DIP.py + Flask)
echo ===============================================================
echo   Starting 100%% Offline Vehicle Detection Server...
echo   Once started, open http://127.0.0.1:5000 in your browser
echo ===============================================================
start "" "http://127.0.0.1:5000"
python app.py
pause
