@echo off
REM Instances a donnees reelles Azure-trace 100x10 et 200x10 (Tableaux 12 et A4). Demandes deja fournies dans noyau\data
REM Duree indicative : 4-6 h. Si le calcul s arrete, relancer ce fichier : les runs deja faits sont sautes.
cd /d "%~dp0"
set /a W=%NUMBER_OF_PROCESSORS%-1
if %W% LSS 1 set W=1
python ..\noyau\run_v2.py --preset azure --workers %W% --out ..\resultats\results_azure
python ..\noyau\analyze_v2.py --root ..\resultats\results_azure --ref A8_adaptfoot
echo.
echo ===== Termine : voir resultats\results_azure\_analysis =====
pause
