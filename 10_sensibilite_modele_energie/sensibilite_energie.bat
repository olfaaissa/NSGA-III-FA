@echo off
REM Sensibilite des conclusions aux parametres energetiques (PUE, part decarbonee)
REM 2 instances (CloudBroker 50x5, Azure-trace 100x10) x 4 scenarios x 3 methodes x 30 runs
REM Prerequis conseille : etapes 03 et 04 (le scenario de base est copie depuis leurs resultats)
REM Duree indicative : environ 7 a 9 h (une nuit). Si le calcul s arrete, relancer ce fichier : les runs deja faits sont sautes.
cd /d "%~dp0"
set /a W=%NUMBER_OF_PROCESSORS%-1
if %W% LSS 1 set W=1
python sensibilite_energie.py --workers %W% --runs 30
python sensibilite_energie.py --analyze
echo.
echo ===== Termine : voir resultats\results_sens_energie\_analysis =====
pause
