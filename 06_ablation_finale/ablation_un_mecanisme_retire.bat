@echo off
REM Ablation un mecanisme retire a la fois L1-L8 (Tableaux 14 et A3). Prerequis : etapes 03 et 05
REM Duree indicative : 6-7 h. Si le calcul s arrete, relancer ce fichier : les runs deja faits sont sautes.
cd /d "%~dp0"
set /a W=%NUMBER_OF_PROCESSORS%-1
if %W% LSS 1 set W=1
python ablation_finale.py --workers %W% --runs 30
python ..\noyau\analyze_v2.py --root ..\resultats\results_ablation_final --ref A8_adaptfoot
echo.
echo ===== Termine : voir resultats\results_ablation_final\_analysis =====
pause
