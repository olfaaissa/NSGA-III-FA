@echo off
REM Comparaison avec NSGA-III-GKM++ sur des instances plus grandes (50x5, 100x10, 200x10)
REM 4 methodes x 3 instances x 30 runs, 125 000 evaluations
REM Duree indicative : environ 2 a 3 h. Si le calcul s arrete, relancer ce fichier : les runs deja faits sont sautes.
cd /d "%~dp0"
set /a W=%NUMBER_OF_PROCESSORS%-1
if %W% LSS 1 set W=1
python comparaison_gkm_grandes.py --workers %W% --runs 30
python comparaison_gkm_grandes.py --analyze
echo.
echo ===== Termine : voir resultats\results_gkm_grandes\_analysis =====
pause
