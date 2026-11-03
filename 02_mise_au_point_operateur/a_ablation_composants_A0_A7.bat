@echo off
REM Ablation par composant A0-A7 et sensibilite de l empreinte (Tableaux A1 et A2 de l article)
REM Duree indicative : 8-12 h. Si le calcul s arrete, relancer ce fichier : les runs deja faits sont sautes.
cd /d "%~dp0"
set /a W=%NUMBER_OF_PROCESSORS%-1
if %W% LSS 1 set W=1
python ..\noyau\run_v2.py --preset ablation --workers %W% --out ..\resultats\results_ablation
python ..\noyau\run_v2.py --preset varprob --workers %W% --out ..\resultats\results_varprob
python ..\noyau\analyze_v2.py --root ..\resultats\results_ablation
python ..\noyau\analyze_v2.py --root ..\resultats\results_varprob
echo.
echo ===== Termine : voir resultats\results_ablation et results_varprob =====
pause
