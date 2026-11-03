@echo off
REM Figures de convergence de l article (Fig. 4, A1 et A2). Prerequis : etapes 03 et 05
REM Duree indicative : quelques minutes. Si le calcul s arrete, relancer ce fichier : les runs deja faits sont sautes.
cd /d "%~dp0"
set /a W=%NUMBER_OF_PROCESSORS%-1
if %W% LSS 1 set W=1
python ..\noyau\analyze_v2.py --root ..\resultats\results_cloud --ref A8_adaptfoot --only cb_N50_M5_M4 cb_N200_M10_M4 --out ..\resultats\results_cloud\_figs_article
python ..\noyau\analyze_v2.py --root ..\resultats\results_dtlz --ref A8_adaptfoot --only dtlz7_M10 dtlz3_M5 --out ..\resultats\results_dtlz\_figs_article
echo.
echo ===== Termine : voir resultats\results_cloud\_figs_article =====
pause
