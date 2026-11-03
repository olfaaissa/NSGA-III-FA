@echo off
REM Comparaison avec NSGA-III-GKM++ sur la formulation harmonisee (Tableau 18), avec la variante GKM++ a initialisation completee
REM Duree indicative : environ 30 min. Si le calcul s arrete, relancer ce fichier : les runs deja faits sont sautes.
cd /d "%~dp0"
set /a W=%NUMBER_OF_PROCESSORS%-1
if %W% LSS 1 set W=1
python comparaison_gkm.py --workers %W% --out ..\resultats\results_gkm
python analyse_gkm.py --root ..\resultats\results_gkm
echo.
echo ===== Termine : voir resultats\results_gkm\_analysis =====
pause
