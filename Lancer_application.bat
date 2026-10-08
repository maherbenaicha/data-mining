@echo off
title Crises d'absence - EEG
cd /d "%~dp0"
echo.
echo  Lancement de l'application... Le navigateur va s'ouvrir tout seul.
echo  Pour ARRETER l'application : ferme cette fenetre noire.
echo.
python -m streamlit run app.py --server.headless false --browser.gatherUsageStats false
pause
