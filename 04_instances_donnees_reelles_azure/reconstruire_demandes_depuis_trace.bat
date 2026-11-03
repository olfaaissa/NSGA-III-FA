@echo off
REM FACULTATIF : retelecharge la trace Azure (440 Mo) et reconstruit noyau\data (resultat identique)
REM Duree indicative : 5-10 min. Si le calcul s arrete, relancer ce fichier : les runs deja faits sont sautes.
cd /d "%~dp0"
set /a W=%NUMBER_OF_PROCESSORS%-1
if %W% LSS 1 set W=1
python construire_instances_azure.py
echo.
echo ===== Termine : voir noyau\data =====
pause
