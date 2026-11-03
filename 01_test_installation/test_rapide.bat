@echo off
REM Test rapide de l installation (2 runs tres courts, quelques minutes)
REM Duree indicative : 2-5 min. Si le calcul s arrete, relancer ce fichier : les runs deja faits sont sautes.
cd /d "%~dp0"
set /a W=%NUMBER_OF_PROCESSORS%-1
if %W% LSS 1 set W=1
python ..\noyau\run_v2.py --problems dtlz2 --n_obj 5 --algos NSGA3-PM A5_FA2 --runs 2 --fe_scale 0.03 --out ..\resultats\test_rapide
echo.
echo ===== Termine : voir resultats\test_rapide =====
pause
