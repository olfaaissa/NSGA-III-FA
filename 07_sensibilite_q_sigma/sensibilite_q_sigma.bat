@echo off
REM Sensibilite du seuil adaptatif : grille 5x5 de (q, sigma), CloudBroker 50x5 (Fig. A3). Prerequis : etape 03
REM Duree indicative : environ 6 h. Si le calcul s arrete, relancer ce fichier : les runs deja faits sont sautes.
cd /d "%~dp0"
set /a W=%NUMBER_OF_PROCESSORS%-1
if %W% LSS 1 set W=1
python sensibilite_q_sigma.py --workers %W% --runs 30
python sensibilite_q_sigma.py --analyze
echo.
echo ===== Termine : voir resultats\results_sens_qsigma\_analysis =====
pause
