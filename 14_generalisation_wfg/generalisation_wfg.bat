@echo off
REM Generalisation WFG : WFG1-2 et WFG4-9, M = 3, 5, 8, 6 methodes, 30 runs (memes methodes que l etape 05)
REM Duree indicative : environ 10 a 15 h. Si le calcul s arrete, relancer ce fichier : les runs deja faits sont sautes.
cd /d "%~dp0"
set /a W=%NUMBER_OF_PROCESSORS%-1
if %W% LSS 1 set W=1
python ..\noyau\run_v2.py --preset wfg_gen --workers %W% --out ..\resultats\results_wfg
python ..\noyau\analyze_v2.py --root ..\resultats\results_wfg --ref A8_adaptfoot
echo.
echo ===== Termine : voir resultats\results_wfg\_analysis =====
pause
