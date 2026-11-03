@echo off
REM Consolidation des serveurs et serveurs recents (SPECpower 2024)
REM 2 instances x 3 scenarios x 3 methodes x 30 runs (le scenario de base est copie depuis l etape 10)
REM Duree indicative : environ 3 a 5 h. Si le calcul s arrete, relancer ce fichier : les runs deja faits sont sautes.
cd /d "%~dp0"
set /a W=%NUMBER_OF_PROCESSORS%-1
if %W% LSS 1 set W=1
python consolidation_serveurs_recents.py --workers %W% --runs 30
python consolidation_serveurs_recents.py --analyze
echo.
echo ===== Termine : voir resultats\results_consolidation_recents\_analysis =====
pause
