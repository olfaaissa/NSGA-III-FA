@echo off
REM Sensibilite du seuil (q, sigma) sur 3 instances jamais utilisees pour un reglage
REM Azure-trace 100x10 (grille 3x3), DTLZ2 M=8 et DTLZ4 M=5 (grilles 5x5), 30 runs par case
REM Prerequis conseille : etapes 04 et 05 (la case par defaut est copiee depuis leurs resultats)
REM Duree indicative : environ 6 a 8 h (une nuit). Si le calcul s arrete, relancer ce fichier : les runs deja faits sont sautes.
cd /d "%~dp0"
set /a W=%NUMBER_OF_PROCESSORS%-1
if %W% LSS 1 set W=1
python sensibilite_q_sigma_autres.py --workers %W% --runs 30
python sensibilite_q_sigma_autres.py --analyze
echo.
echo ===== Termine : voir resultats\results_sens_qsigma_autres\_analysis =====
pause
