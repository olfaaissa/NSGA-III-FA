"""
operators_v2.py -- Mutation Firefly guidee et adaptative de NSGA-III-FA, et
variantes d'ablation.

Une seule classe, FireflyMutationV2, reproduit toutes les variantes de
l'ablation grace a des drapeaux. Ainsi, deux variantes ne different
que par le drapeau qu'on change.

Principes de l'operateur (etiquettes reprises dans le code) :

  [OP-1] ZERO evaluation supplementaire. L'operateur n'utilise que les
         fitness DEJA connues de la population parente (algorithm.pop),
         evaluees par la selection de NSGA-III : aucun appel a
         problem._evaluate, donc le budget compte par pymoo est le budget reel.
  [OP-2] Dispersion normalisee. Chaque objectif est normalise avec le point
         ideal et le point nadir que NSGA-III maintient deja
         (HyperplaneNormalization) : delta_t est sans unite, a peu pres dans
         [0 ; 0,5]. Sur les objectifs bruts, l'energie (~200) ecraserait le
         temps de reponse.
  [OP-3] Seuil adaptatif. Avant W0 observations, on reste en regime
         exploration ; ensuite theta_t = Quantile_Q(H) + sigma * std(H).
  [OP-4] Operateur toujours actif, avec cet ordre de repli : (a) un parent qui
         domine le proxy, tire AU HASARD ; (b) sinon un parent meilleur dans la
         meme niche de reference NSGA-III ; (c) sinon un parent du premier front.
  [OP-5] Pas de dependance a l'ordre. L'attracteur est tire au hasard parmi les
         candidats, et les attracteurs sont des parents fixes : le resultat ne
         depend pas de l'ordre de parcours.
  [OP-6] Pas alpha a l'echelle des variables, avec une decroissance indexee
         sur le budget consomme (t_eq = n_eval / N) et non sur n_gen.
         Distance r normalisee par (xu - xl), ce qui rend gamma sans unite.
  [OP-8] Empreinte de mutation appariee a PM. Chaque variable est perturbee
         avec la probabilite 1/n, comme PM. FA et PM different donc uniquement
         par la direction (attraction) et l'amplitude du pas, pas par le nombre
         de variables touchees. Essai preliminaire (6 runs, 15 % du budget) :
         vecteur entier -> IGD+ DTLZ2 M=5 : 0,249 ; 1/n -> 0,119 ; PM : 0,104.
         La variante A7 perturbe le vecteur entier.
  [OP-9] Empreinte adaptative (option footprint) : le regime choisi par la
         dispersion fixe la part des variables perturbees (large en
         exploration, 1/n en exploitation). Variantes A8, A9.
  [OP-10] Plancher alpha_min du pas aleatoire (A10, A11) : sans plancher, la
         decroissance geometrique eteint l'exploration sur les problemes
         multimodaux (DTLZ1, DTLZ3).
  [OP-11] beta0 = 1 (A12, A13) au lieu de 2 : evite que l'attraction depasse
         l'attracteur et piege la population sur un front local (DTLZ3).
  [OP-7] Journal d'activation par generation : taux d'activation, part
         dominant / repli niche / repli elite / inactif, regime, delta brut ET
         normalise (pour la figure brut contre normalise).

Ablation -- drapeaux :
    guided      : attraction vers un attracteur (sinon pas aleatoire seul)
    decay       : decroissance de alpha
    switching   : bascule exploration/exploitation par dispersion
    normalize   : dispersion normalisee (False = objectifs bruts)
    fallback    : repli niche/elite si aucun dominant (False = dominant seul)
    var_prob    : part des variables perturbees (0 => 1/n ; 1 => vecteur entier)
    local, beta0, alpha0 : options pour l'analyse de sensibilite
"""

from collections import deque

import numpy as np
from pymoo.core.mutation import Mutation
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


# Configurations d'ablation. A5 = methode proposee.
ABLATIONS = {
    "A0_randomwalk": dict(guided=False, decay=True,  switching=False, normalize=True,  fallback=True),
    "A1_FC":         dict(guided=True,  decay=False, switching=False, normalize=True,  fallback=True),
    "A2_decay":      dict(guided=True,  decay=True,  switching=False, normalize=True,  fallback=True),
    "A3_switch":     dict(guided=True,  decay=False, switching=True,  normalize=True,  fallback=True),
    "A4_rawdisp":    dict(guided=True,  decay=True,  switching=True,  normalize=False, fallback=True),
    "A5_FA2":        dict(guided=True,  decay=True,  switching=True,  normalize=True,  fallback=True),
    "A6_nofallback": dict(guided=True,  decay=True,  switching=True,  normalize=True,  fallback=False),
    # [OP-8] empreinte "vecteur entier" (comme NSGA3-PMm)
    "A7_fullvector": dict(guided=True,  decay=True,  switching=True,  normalize=True,  fallback=True, var_prob=1.0),
}
# Toutes les variantes utilisent var_prob = 1/n (defaut), sauf A7.

# Sensibilite a la part de variables perturbees (compromis observe lors des essais preliminaires :
# vecteur entier = meilleur sur CloudBroker, pire sur DTLZ). A5 = 1/n, A7 = 1.0.
# [OP-9] Mecanisme adaptatif enrichi (hypothese issue de l'ablation A0-A7) :
# la bascule par dispersion choisit l'empreinte de mutation : large (0,5) en
# exploration, fine (1/n) en exploitation. A9 : idem, sans decroissance de alpha,
# qui penalisait CloudBroker.
ABLATIONS["A8_adaptfoot"] = dict(guided=True, decay=True, switching=True, normalize=True,
                                 fallback=True, footprint=(0.5, 0.0))
# [OP-10] Essai preliminaire : sur DTLZ1/DTLZ3 (multimodaux, jamais vus), toutes les
# variantes avec decroissance s'effondrent (IGD+ x5 a x50 vs PM) car
# alpha = 1.5*0.95^t vaut ~0.009 apres 100 generations : plus d'exploration.
# Correctif : plancher du pas, a l'echelle d'une mutation polynomiale (eta=20).
ABLATIONS["A10_floor05"] = dict(guided=True, decay=True, switching=True, normalize=True,
                                fallback=True, footprint=(0.5, 0.0), alpha_min=0.05)
ABLATIONS["A11_floor10"] = dict(guided=True, decay=True, switching=True, normalize=True,
                                fallback=True, footprint=(0.5, 0.0), alpha_min=0.10)
# [OP-11] beta0 = 1 (valeur canonique de Yang 2008) au lieu de 2 : avec
# beta0 = 2, le pas d'attraction depasse l'attracteur (x2) ; mini-test DTLZ3
# M=5 (3 runs, 30 % du budget) : IGD+ 3,48 (beta0 = 2) -> 0,73 (beta0 = 1).
ABLATIONS["A12_beta1"] = dict(guided=True, decay=True, switching=True, normalize=True,
                              fallback=True, footprint=(0.5, 0.0), beta0=1.0)
ABLATIONS["A13_beta1_nodecay"] = dict(guided=True, decay=False, switching=True, normalize=True,
                                      fallback=True, footprint=(0.5, 0.0), beta0=1.0)
ABLATIONS["A9_adaptfoot_nodecay"] = dict(guided=True, decay=False, switching=True, normalize=True,
                                         fallback=True, footprint=(0.5, 0.0))

# [OP-12] Ablation "un mecanisme retire a la fois" de la methode finale A8.
# Chaque variante = A8 avec UN seul mecanisme desactive ; tout le reste identique.
_A8 = ABLATIONS["A8_adaptfoot"]
ABLATIONS["L1_noattraction"] = dict(_A8, guided=False)              # pas aleatoire, sans attraction
ABLATIONS["L2_nodecay"]      = dict(_A8, decay=False)               # pas constant
ABLATIONS["L3_noswitch"]     = dict(_A8, switching=False)           # aucun regime : empreinte 0,5 et parametres fixes
ABLATIONS["L4_fixedparams"]  = dict(_A8, adapt_params=False)        # empreinte adaptative, beta0/gamma/p/pas fixes
ABLATIONS["L5_fixedfoot"]    = dict(_A8, footprint=(0.5, 0.5))      # parametres adaptatifs, empreinte fixe 0,5
ABLATIONS["L6_nofallback"]   = dict(_A8, fallback=False)            # dominant seul, sans repli
ABLATIONS["L7_rawdisp"]      = dict(_A8, normalize=False)           # dispersion sur objectifs bruts
ABLATIONS["L8_firstfound"]   = dict(_A8, first=True)                # parcours sequentiel (premier trouve) au lieu du tirage

for _vp in (0.1, 0.3, 0.5):
    ABLATIONS[f"S_vp{_vp}"] = dict(guided=True, decay=True, switching=True,
                                   normalize=True, fallback=True, var_prob=_vp)


class FireflyMutationV2(Mutation):

    def __init__(self, guided=True, decay=True, switching=True, normalize=True,
                 fallback=True, local=False, var_prob=0.0, footprint=None, alpha_min=1e-6,
                 alpha0=1.5, b_alpha=0.95, beta0=2.0, gamma=0.1, prob0=0.7,
                 window=50, warmup=20, q=0.15, sigma=0.5, seed=None,
                 first=False, adapt_params=True, **kwargs):
        # prob=1.0 au niveau de pymoo : la probabilite par individu (prob0
        # ou 2*prob0) est geree ici.
        super().__init__(prob=1.0, **kwargs)
        self.guided, self.decay, self.switching = guided, decay, switching
        self.normalize, self.fallback, self.local = normalize, fallback, local
        # [OP-12] ablation finale : first=True reproduit un parcours
        # sequentiel (premier candidat trouve, dependance a l'ordre) ;
        # adapt_params=False fige beta0, gamma, p et le pas aux valeurs
        # d'exploration (seule l'empreinte suit encore le regime).
        self.first, self.adapt_params = first, adapt_params
        self.var_prob = var_prob            # [OP-8] 0 => 1/n ; 1.0 => tout le vecteur
        # [OP-9] empreinte adaptative : (var_prob en exploration, var_prob en
        # exploitation), choisie par le regime de dispersion. None = fixe.
        self.footprint = footprint
        self._vp_now = var_prob
        self.alpha_min = alpha_min          # [OP-10] plancher du pas aleatoire
        # [OP-9] empreinte adaptative : (var_prob en exploration, var_prob en
        # exploitation). None = empreinte fixe var_prob.
        self.alpha0, self.b_alpha = alpha0, b_alpha
        self.beta0, self.gamma, self.prob0 = beta0, gamma, prob0
        self.warmup, self.q, self.sigma = warmup, q, sigma
        self.history = deque(maxlen=window)
        self.log = []                      # [OP-7]
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _ideal_nadir(algorithm, F):
        """[OP-2] Reutilise la normalisation de NSGA-III. Si elle est absente
        (autre algorithme), on prend min/max de la population."""
        ideal = nadir = None
        surv = getattr(algorithm, "survival", None)
        norm = getattr(surv, "norm", None)
        if norm is not None:
            ideal = getattr(norm, "ideal_point", None)
            nadir = getattr(norm, "nadir_point", None)
        if ideal is None or nadir is None:
            ideal, nadir = F.min(axis=0), F.max(axis=0)
        span = np.where(np.abs(nadir - ideal) < 1e-12, 1.0, nadir - ideal)
        return ideal, span

    @staticmethod
    def _dominance_matrix(F):
        """D[j, p] = True si le parent j domine le parent p (vectorise)."""
        le = np.all(F[:, None, :] <= F[None, :, :], axis=2)
        lt = np.any(F[:, None, :] < F[None, :, :], axis=2)
        return le & lt

    def _var_mask(self, rng, n_var):
        """[OP-8] Variables perturbees : chacune avec la probabilite var_prob
        (0 => 1/n, comme PM), au moins une. var_prob = 1 => vecteur entier."""
        if self._vp_now >= 1.0:
            return np.ones(n_var)
        vp = (1.0 / n_var) if self._vp_now <= 0 else self._vp_now
        mask = rng.random(n_var) < vp
        if not mask.any():
            mask[rng.integers(n_var)] = True
        return mask.astype(float)

    # ------------------------------------------------------------------- core
    def _do(self, problem, X, *args, random_state=None, algorithm=None, **kwargs):
        rng = random_state if random_state is not None else self._rng
        xl, xu = np.asarray(problem.xl, float), np.asarray(problem.xu, float)
        rngx = np.where((xu - xl) > 0, xu - xl, 1.0)
        n_off, n_var = X.shape
        Y = X.copy()

        # ---- [OP-1] information parente, deja evaluee : 0 FE ----
        pop = algorithm.pop
        PX, PF = pop.get("X"), pop.get("F")
        n_par = len(PX)
        N_ref = getattr(algorithm, "pop_size", None) or n_par
        n_eval = algorithm.evaluator.n_eval
        t_eq = n_eval / float(N_ref)        # [OP-6] "generations equivalentes"

        # ---- [OP-2] dispersion ----
        ideal, span = self._ideal_nadir(algorithm, PF)
        delta_raw = float(np.mean(np.std(PF, axis=0)))
        delta_norm = float(np.mean(np.std((PF - ideal) / span, axis=0)))
        delta = delta_norm if self.normalize else delta_raw
        self.history.append(delta)

        # ---- [OP-3] regime ----
        theta = np.nan
        regime = "explore"
        if self.switching and len(self.history) >= self.warmup:
            H = np.asarray(self.history)
            theta = float(np.quantile(H, self.q) + self.sigma * np.std(H))
            if delta < theta:
                regime = "exploit"

        # [OP-9] le regime pilote aussi l'empreinte de mutation
        if self.footprint is not None:
            self._vp_now = self.footprint[0] if regime == "explore" else self.footprint[1]
        else:
            self._vp_now = self.var_prob
        # [OP-9] la dispersion pilote aussi l'empreinte de mutation
        alpha = self.alpha0 * (self.b_alpha ** t_eq) if self.decay else self.alpha0
        alpha = max(alpha, self.alpha_min)  # [OP-10] plancher du pas
        if regime == "exploit" and self.adapt_params:   # multiplicateurs du regime exploitation
            beta0, prob, gamma, alpha_l = 0.5 * self.beta0, min(1.0, 2 * self.prob0), 0.5 * self.gamma, 0.8 * alpha
        else:
            beta0, prob, gamma, alpha_l = self.beta0, self.prob0, self.gamma, (1.2 * alpha if (self.switching and self.adapt_params) else alpha)

        # ---- structures pour le choix de l'attracteur ----
        rank = pop.get("rank")
        niche = pop.get("niche")
        dist_niche = pop.get("dist_to_niche")
        if rank is None or np.any([r is None for r in np.atleast_1d(rank)]):
            rank = np.zeros(n_par, dtype=int)
            for k, front in enumerate(NonDominatedSorting().do(PF)):
                rank[front] = k
        rank = np.asarray(rank, dtype=float)
        has_niche = niche is not None and not np.any([v is None for v in np.atleast_1d(niche)])
        if has_niche:
            niche = np.asarray(niche, dtype=int)
            dist_niche = np.asarray(dist_niche, dtype=float)
        D = self._dominance_matrix(PF) if self.guided else None
        elite = np.flatnonzero(rank == rank.min())

        # proxy de chaque descendant = parent le plus proche (espace des
        # decisions normalise) : il remplace la fitness du descendant, que
        # l'on n'evalue plus [OP-1].
        Xn, PXn = X / rngx, PX / rngx
        # ||a-b||^2 = |a|^2 + |b|^2 - 2ab : memoire N x N (et non N x N x n_var)
        d2 = (Xn ** 2).sum(1)[:, None] + (PXn ** 2).sum(1)[None, :] - 2.0 * Xn @ PXn.T
        proxy = d2.argmin(axis=1)

        c = dict(selected=0, dom=0, niche=0, elite=0, inactive=0, randomwalk=0)
        for i in range(n_off):
            if rng.random() > prob:
                continue
            c["selected"] += 1
            noise = alpha_l * rngx * (rng.random(n_var) - 0.5)

            if not self.guided:                       # A0 : pas aleatoire seul
                Y[i] = np.clip(Y[i] + noise * self._var_mask(rng, n_var), xl, xu)
                c["randomwalk"] += 1
                continue

            p = proxy[i]
            j, kind = None, None
            doms = np.flatnonzero(D[:, p])
            # local=True : candidat le plus proche ; sinon tirage uniforme [OP-5]
            if self.first:                            # [OP-12] premier trouve
                pick = lambda C: C[0]
            elif self.local:
                pick = lambda C: C[np.argmin(d2[i, C])]
            else:
                pick = lambda C: rng.choice(C)
            if len(doms) > 0:                         # [OP-4a] + [OP-5]
                j, kind = pick(doms), "dom"
            elif self.fallback:
                if has_niche:                         # [OP-4b] meme niche
                    same = np.flatnonzero((niche == niche[p]) &
                                          ((rank < rank[p]) |
                                           ((rank == rank[p]) & (dist_niche < dist_niche[p]))))
                    if len(same) > 0:
                        j, kind = pick(same), "niche"
                if j is None:                         # [OP-4c] premier front
                    cand = elite[elite != p]
                    if len(cand) > 0:
                        j, kind = pick(cand), "elite"
            if j is None:                             # sans repli : inactif
                c["inactive"] += 1
                continue

            r = np.linalg.norm((Y[i] - PX[j]) / rngx)  # [OP-6] distance normalisee
            beta = beta0 * np.exp(-gamma * r ** 2)
            step = (beta * (PX[j] - Y[i]) + noise) * self._var_mask(rng, n_var)   # [OP-8]
            Y[i] = np.clip(Y[i] + step, xl, xu)
            c[kind] += 1

        moved = c["dom"] + c["niche"] + c["elite"] + c["randomwalk"]
        self.log.append(dict(
            n_eval=int(n_eval), t_eq=round(t_eq, 3), delta_raw=delta_raw, delta_norm=delta_norm,
            theta=theta, regime=regime, alpha=alpha_l, prob=prob, var_prob=self._vp_now, n_off=n_off,
            selected=c["selected"], moved=moved, dom=c["dom"], niche=c["niche"],
            elite=c["elite"], randomwalk=c["randomwalk"], inactive=c["inactive"],
            activation_rate=moved / float(n_off),
        ))
        return Y
