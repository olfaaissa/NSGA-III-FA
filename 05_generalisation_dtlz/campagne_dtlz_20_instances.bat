@echo off
REM Generalisation : DTLZ1-4 et DTLZ7, M = 3, 5, 8, 10, 6 methodes (Tableaux 16, A5 et A6, Fig. A1 et A2)
REM Duree indicative : 1-2 jours. Si le calcul s arrete, relancer ce fichier : les runs deja faits sont sautes.
cd /d "%~dp0"
set /a W=%NUMBER_OF_PROCESSORS%-1
if %W% LSS 1 set W=1
python ..\noyau\run_v2.py --preset dtlz_gen --workers %W% --out ..\resultats\results_dtlz
python ..\noyau\analyze_v2.py --root ..\resultats\results_dtlz --ref A8_adaptfoot
echo.
echo ===== Termine : voir resultats\results_dtlz\_analysis =====
pause
