@echo off
REM Tableaux LaTeX et figures dispersion / SPECpower de l article (Fig. 2 et 5). Prerequis : etapes 02, 03 et 05
REM Duree indicative : 1 min. Si le calcul s arrete, relancer ce fichier : les runs deja faits sont sautes.
cd /d "%~dp0"
set /a W=%NUMBER_OF_PROCESSORS%-1
if %W% LSS 1 set W=1
python generer_tableaux_figures.py
echo.
echo ===== Termine : voir article_tableaux_figures =====
pause
