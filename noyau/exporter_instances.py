"""
exporter_instances.py -- Exporte les donnees completes des six instances de courtage
(4 CloudBroker generees + 2 Azure-trace) dans des fichiers CSV, pour que l'energie
de n'importe quelle allocation puisse etre recalculee a partir des valeurs publiees.

Pour chaque instance <nom> :
  instances/<nom>_fournisseurs.csv : j, type, serveur, kappa_j (unites/serveur),
      A_j (unites), S_j = A_j/kappa_j (serveurs, nombre reel), T_j, C_j, PUE_j, r_j
  instances/<nom>_clients.csv      : i, R_i (unites), P_i
  instances/<nom>_latences.csv     : matrice L_ij (N lignes x M colonnes)

Usage : python exporter_instances.py   (ecrit dans <racine>/instances/)
"""
import os, sys
import numpy as np, pandas as pd
_ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ICI)
from cloud_v2 import ScaledCloudBroker

OUT = os.path.join(os.path.dirname(_ICI), "instances")
INSTANCES = [("cb_N10_M3", 10, 3, "cb"), ("cb_N50_M5", 50, 5, "cb"), ("cb_N100_M10", 100, 10, "cb"),
             ("cb_N200_M10", 200, 10, "cb"), ("az_N100_M10", 100, 10, "az"), ("az_N200_M10", 200, 10, "az")]
TYPE = {"HP_ML110_G4_Xeon3040": "Small", "HP_ML110_G5_Xeon3075": "Medium", "IBM_x3550_XeonX5670": "XL"}

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for name, N, M, kind in INSTANCES:
        p = ScaledCloudBroker(N, M, kind=kind)
        pd.DataFrame({"j": np.arange(1, M + 1), "type": [TYPE[s] for s in p.servers], "serveur": p.servers,
                      "kappa_j": p.units_per_server, "A_j": p.capacities, "S_j": p.capacities / p.units_per_server,
                      "T_j": p.T, "C_j": p.C, "PUE_j": p.pue, "r_j": p.r}).to_csv(
            os.path.join(OUT, f"{name}_fournisseurs.csv"), index=False, float_format="%.6g")
        pd.DataFrame({"i": np.arange(1, N + 1), "R_i": p.demands, "P_i": p.P}).to_csv(
            os.path.join(OUT, f"{name}_clients.csv"), index=False, float_format="%.6g")
        np.savetxt(os.path.join(OUT, f"{name}_latences.csv"), p.L, delimiter=",", fmt="%.6g")
        print(f"{name}: demande totale {p.demands.sum():.0f}, capacite totale {p.capacities.sum():.2f}, "
              f"S_j = {np.round(p.capacities / p.units_per_server, 2).tolist()}")
