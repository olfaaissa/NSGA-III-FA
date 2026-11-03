"""
make_azure_instance.py -- [AZ-1] Reconstruit les demandes clients des instances
az_N100_M10 et az_N200_M10 a partir de la trace Azure Public Dataset V2 (2019).

Les fichiers produits (data/azure2019_h215_N100.csv, ..._N200.csv) sont deja
fournis : ce script sert seulement a la reproductibilite (telechargement ~440 Mo).

Source : https://github.com/Azure/AzurePublicDataset (AzurePublicDatasetV2.md)
  Cortez et al., "Resource Central: Understanding and predicting workloads for
  improved resource management in large cloud platforms", SOSP 2017.
Regle de selection :
  1. on ignore l'heure 0 (VM deja presentes au debut de la trace) ;
  2. heure h = celle dont le nombre d'abonnements creant des VM est la mediane
     des heures 1..719 (premiere en cas d'egalite) -> h = 215 ;
  3. client = abonnement ; R_i = somme des vCPU des VM creees pendant h
     (bucket ">24" compte 30 vCPU) ;
  4. echantillon aleatoire de N abonnements (random_state = 2026).
Usage : python construire_instances_azure.py
"""
# --- chemins du dossier organise : le noyau partage et le dossier resultats/ ---
import os as _os, sys as _sys
_ICI = _os.path.dirname(_os.path.abspath(__file__))
_RACINE = _os.path.dirname(_ICI)
_sys.path.insert(0, _os.path.join(_RACINE, "noyau"))
RESULTATS = _os.path.join(_RACINE, "resultats")
# ---------------------------------------------------------------------------
import os
import urllib.request

import pandas as pd

URL = ("https://github.com/Azure/AzurePublicDataset/releases/download/"
       "dataset-v2/trace_data_vmtable_vmtable.csv.gz")
HERE = os.path.dirname(os.path.abspath(__file__))
COLS = ["vmid", "sub", "dep", "t0", "t1", "maxcpu", "avgcpu", "p95", "cat", "cores", "mem"]


def main():
    path = os.path.join(HERE, "vmtable.csv.gz")
    if not os.path.exists(path):
        print("Telechargement de la trace Azure (~440 Mo)...")
        urllib.request.urlretrieve(URL, path)
    df = pd.read_csv(path, header=None, names=COLS, usecols=["sub", "t0", "cores", "cat"])
    df["c"] = df["cores"].replace(">24", "30").astype(float)
    df = df[df["t0"] >= 3600].copy()
    df["h"] = df["t0"] // 3600
    nsub = df.groupby("h")["sub"].nunique()
    h = int((nsub - nsub.median()).abs().sort_values(kind="stable").index[0])
    w = df[df["h"] == h]
    per = w.groupby("sub").agg(R=("c", "sum"), nvm=("c", "size"),
                               cats=("cat", lambda s: s.mode().iat[0])).reset_index()
    DATA = os.path.join(_RACINE, "noyau", "data")   # lu par noyau/cloud_v2.py
    os.makedirs(DATA, exist_ok=True)
    for n in (100, 200):
        s = per.sample(n=n, random_state=2026).reset_index(drop=True)
        out = os.path.join(DATA, f"azure2019_h{h}_N{n}.csv")
        s[["R", "nvm", "cats"]].to_csv(out, index_label="client")
        print(f"heure {h} : {len(per)} abonnements actifs ; N={n}, demande totale {s.R.sum():.0f} vCPU -> {out}")


if __name__ == "__main__":
    main()
