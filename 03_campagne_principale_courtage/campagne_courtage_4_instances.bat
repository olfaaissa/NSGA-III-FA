@echo off
REM CAMPAGNE PRINCIPALE : 4 instances CloudBroker, 9 methodes, 30 runs (Tableaux 9-11 et 15, Fig. 4, 5 et A1)
REM Duree indicative : 5-8 h. Si le calcul s arrete, relancer ce fichier : les runs deja faits sont sautes.
cd /d "%~dp0"
set /a W=%NUMBER_OF_PROCESSORS%-1
if %W% LSS 1 set W=1
python ..\noyau\run_v2.py --preset cloud --workers %W% --out ..\resultats\results_cloud
python ..\noyau\analyze_v2.py --root ..\resultats\results_cloud --ref A8_adaptfoot
echo.
echo ===== Termine : voir resultats\results_cloud\_analysis =====
pause
