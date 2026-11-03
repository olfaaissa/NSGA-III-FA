@echo off
REM Choix de la methode finale A8 : dispersion adaptative, puis instances jamais vues (Tableau 13)
REM Duree indicative : 10-14 h. Si le calcul s arrete, relancer ce fichier : les runs deja faits sont sautes.
cd /d "%~dp0"
set /a W=%NUMBER_OF_PROCESSORS%-1
if %W% LSS 1 set W=1
python ..\noyau\run_v2.py --preset adaptive --workers %W% --out ..\resultats\results_adaptive
python ..\noyau\run_v2.py --preset adaptive2 --workers %W% --out ..\resultats\results_adaptive
python ..\noyau\analyze_v2.py --root ..\resultats\results_adaptive
echo.
echo ===== Termine : voir resultats\results_adaptive =====
pause
