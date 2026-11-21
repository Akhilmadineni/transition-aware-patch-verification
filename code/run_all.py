#!/usr/bin/env python3
"""A5P4 - Transition-aware verification of AI-generated infrastructure security patches.

Runs the complete experiment:
  * pre-processing (median imputation + missing flags, winsorising + log of heavy-tailed counts,
    one-hot encoding, standardisation; all fitted on the training split only)
  * transition-aware feature construction (Eq. (1) of the paper)
  * proposed method: distance-weighted k-NN transition verifier (DW-kNN-TV) with a mutual-information
    weighted metric; k and the distance exponent p tuned on the validation split
  * baselines: static final-state policy check, logistic regression, decision tree
    (on raw plan features and on the same transition-aware features)
  * ablation, (k, p) sensitivity grid, prioritised-verification recall@budget, exposure-window regression
  * 5 random stratified 60/20/20 splits (seeds 0-4), mean +- std and paired t-tests
Writes results/*.csv, results/metrics.json, figures/*.png (300 dpi) and results/*.jpg.
"""
import os, json, time, subprocess, shutil, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (f1_score, precision_score, recall_score, average_precision_score,
                             roc_auc_score, precision_recall_curve, confusion_matrix, r2_score,
                             balanced_accuracy_score)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "A5P4_ai_security_patches.csv")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
CLASSES = ["safe", "final_state_insecure", "transient_exposure", "transient_outage",
           "transient_exposure_and_outage"]
CLS_SHORT = ["Safe", "Final-insec.", "Trans. exp.", "Trans. outage", "Exp.+outage"]
K_GRID = [5, 10, 15, 25, 40, 60, 90]
P_GRID = [0.0, 0.5, 1.0, 2.0, 3.0]
A_GRID = [0.0, 0.5, 1.0]   # prior-correction exponent alpha, Eq. (5)
EPS = 1e-6
BUDGETS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]
LOG = []

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 7.5,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "figure.dpi": 100})
COL = {"DW-kNN-TV": "#c0392b", "LR": "#2c7fb8", "DT": "#41ab5d", "LR+TF": "#7fb3d5",
       "DT+TF": "#a1d99b", "Static check": "#555555", "Uniform kNN": "#e59866"}


def log(msg):
    print(msg, flush=True); LOG.append(msg)


# ---------------------------------------------------------------- data + features
CAT = ["finding_type", "generator_model", "environment", "primary_resource_type"]
NUM = ["patch_loc", "resources_changed", "rules_removed", "rules_added", "requires_replacement",
       "create_before_destroy", "plan_destroy_count", "ordering_constraints", "explicit_depends_on",
       "eventual_consistency_score", "live_connections", "dependent_resources", "permission_count_delta",
       "apply_parallelism", "static_final_state_check_pass", "generator_confidence"]
MISS = ["eventual_consistency_score", "live_connections", "generator_confidence"]
HEAVY = ["live_connections", "patch_loc"]
TF = ["repl_no_cbd", "rule_churn", "net_rule_removal", "unordered_parallel", "ec_unordered",
      "live_x_net_removal", "tls_replacement", "iam_overtighten", "destroy_x_dependents"]
# never used as features: patch_id, timestamp, finding_id (identifiers), verification_outcome and
# exposure_window_s (targets / post-replay outcomes)


class Prep:
    """Fit-on-train pre-processing; returns raw and transition-aware design matrices."""

    def fit(self, df):
        self.med = {c: df[c].median() for c in NUM}
        self.cap = {c: df[c].quantile(0.995) for c in HEAVY}
        self.levels = {c: sorted(df[c].unique()) for c in CAT}
        R, T = self._build(df)
        self.mu_r, self.sd_r = R.mean(0), R.std(0) + 1e-9
        self.mu_t, self.sd_t = T.mean(0), T.std(0) + 1e-9
        return self

    def _build(self, df):
        d = df.copy()
        miss = np.column_stack([d[c].isna().astype(float).values for c in MISS])
        for c in NUM:
            d[c] = d[c].fillna(self.med[c])
        for c in HEAVY:
            d[c] = np.log1p(d[c].clip(upper=self.cap[c]))
        oh = np.column_stack([(d[c].values == l).astype(float) for c in CAT for l in self.levels[c]])
        raw = np.column_stack([d[NUM].values.astype(float), miss, oh])
        # transition-aware features, Eq. (1)
        unord = 1.0 - d["explicit_depends_on"].values
        net_rm = np.maximum(d["rules_removed"].values - d["rules_added"].values, 0)
        t = np.column_stack([
            d["requires_replacement"].values * (1 - d["create_before_destroy"].values),
            np.minimum(d["rules_removed"].values, d["rules_added"].values),
            net_rm,
            unord * d["ordering_constraints"].values * np.log1p(d["apply_parallelism"].values),
            unord * d["eventual_consistency_score"].values,
            d["live_connections"].values * (net_rm > 0),
            (df["primary_resource_type"].values == "tls_listener") * d["requires_replacement"].values,
            (df["primary_resource_type"].values == "iam_policy") * np.maximum(-d["permission_count_delta"].values, 0),
            d["plan_destroy_count"].values * d["dependent_resources"].values,
        ])
        return raw, np.column_stack([raw, t])

    def names(self):
        return NUM + [f"miss_{c}" for c in MISS] + [f"{c}={l}" for c in CAT for l in self.levels[c]] + TF

    def transform(self, df):
        R, T = self._build(df)
        return (R - self.mu_r) / self.sd_r, (T - self.mu_t) / self.sd_t


# ---------------------------------------------------------------- DW-kNN transition verifier
def mi_weights(X, y_lab, seed):
    mi = mutual_info_classif(X, y_lab, random_state=seed, n_neighbors=5)
    mi = np.maximum(mi, 0)
    return mi / (mi.sum() + 1e-12) * X.shape[1]   # mean weight 1, Eq. (2)


class DWKNN:
    """Distance-weighted kNN on a feature-weighted metric. Neighbours are queried once at k_max,
    then posteriors for any (k, p) are computed in numpy (Eqs. (3)-(4))."""

    def __init__(self, k_max=max(K_GRID)):
        self.k_max = k_max

    def fit(self, X, y, w):
        self.sw = np.sqrt(w)
        self.nn = NearestNeighbors(n_neighbors=self.k_max, algorithm="auto", n_jobs=1).fit(X * self.sw)
        self.y = np.asarray(y)
        return self

    def neighbours(self, X, exclude_self=False):
        k = self.k_max + (1 if exclude_self else 0)
        d, i = self.nn.kneighbors(X * self.sw, n_neighbors=k)
        if exclude_self:
            d, i = d[:, 1:], i[:, 1:]
        return d, i

    def proba(self, d, i, k, p):
        dk, ik = d[:, :k], i[:, :k]
        w = 1.0 / (dk + EPS) ** p
        P = np.zeros((len(d), len(CLASSES)))
        lab = self.y[ik]
        for c in range(len(CLASSES)):
            P[:, c] = (w * (lab == c)).sum(1)
        return P / P.sum(1, keepdims=True)


# ---------------------------------------------------------------- metrics
def best_threshold(y, s):
    pr, rc, th = precision_recall_curve(y, s)
    f = 2 * pr * rc / (pr + rc + 1e-12)
    j = np.nanargmax(f[:-1]) if len(th) else 0
    return th[j] if len(th) else 0.5


def recall_at_budget(y, s, b, rng):
    n = int(round(b * len(y)))
    order = np.lexsort((rng.random(len(s)), -s))   # random tie-breaking
    return y[order[:n]].sum() / y.sum()


def decide(P5, prior, a):
    """Prior-corrected 5-class decision, Eq. (5)."""
    return (P5 / prior ** a).argmax(1)


def tune_alpha(yva, Pva, prior):
    return max(A_GRID, key=lambda a: f1_score(yva, decide(Pva, prior, a), average="macro"))


def eval_all(y5, P5, s_bin, thr, static_pass, rng, prior=None, a=0.0):
    yb = (y5 != 0).astype(int)
    yhat_b = (s_bin >= thr).astype(int)
    yhat5 = P5.argmax(1) if prior is None else decide(P5, prior, a)
    m = {
        "F1_unsafe": f1_score(yb, yhat_b), "Precision": precision_score(yb, yhat_b, zero_division=0),
        "Recall": recall_score(yb, yhat_b), "PR_AUC": average_precision_score(yb, s_bin),
        "ROC_AUC": roc_auc_score(yb, s_bin), "Macro_F1_5class": f1_score(y5, yhat5, average="macro"),
        "Bal_acc_5class": balanced_accuracy_score(y5, yhat5),
        "Recall_static_pass_unsafe": yhat_b[(static_pass == 1) & (yb == 1)].mean(),
        "FPR_safe": yhat_b[yb == 0].mean(),
        "Recall@20%": recall_at_budget(yb, s_bin, 0.20, rng),
    }
    for c in range(len(CLASSES)):
        m[f"recall_{CLASSES[c]}"] = (yhat5[y5 == c] == c).mean() if c != 0 else (yhat_b[y5 == 0] == 0).mean()
        if c != 0:  # detection rate of each hazard class in the binary view
            m[f"detect_{CLASSES[c]}"] = yhat_b[y5 == c].mean()
    return m, yhat5


# ---------------------------------------------------------------- main loop
def main():
    t0 = time.time()
    df = pd.read_csv(DATA)
    df["y"] = df["verification_outcome"].map({c: i for i, c in enumerate(CLASSES)})
    assert df["y"].notna().all()
    log(f"Loaded {len(df)} rows, {df.shape[1] - 1} columns")

    # dataset summary table
    summ = [["Patches (rows)", f"{len(df):,}"], ["Raw columns", "25"],
            ["Plan-level input features", f"{len(NUM)} numeric + {len(CAT)} categorical"],
            ["Unsafe rate (non-safe outcome)", f"{(df.y != 0).mean() * 100:.1f}%"],
            ["Static check passed", f"{df.static_final_state_check_pass.mean() * 100:.1f}%"],
            ["Unsafe among static-passed", f"{(df.y[df.static_final_state_check_pass == 1] != 0).mean() * 100:.1f}%"]]
    for i, c in enumerate(CLASSES):
        summ.append([f"Class {c}", f"{(df.y == i).sum():,} ({(df.y == i).mean() * 100:.1f}%)"])
    for c in MISS:
        summ.append([f"Missing {c}", f"{df[c].isna().mean() * 100:.1f}%"])
    pd.DataFrame(summ, columns=["Item", "Value"]).to_csv(os.path.join(RES, "table1_dataset_summary.csv"), index=False)
    pd.crosstab(df.generator_model, df.verification_outcome).to_csv(os.path.join(RES, "generator_outcome_counts.csv"))

    rows, abl_rows, sens_rows, budget_rows, reg_rows, cm_store, curves, mi_store, hp_rows = [], [], [], [], [], {}, {}, [], []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        idx = np.arange(len(df))
        tr, te = train_test_split(idx, test_size=0.2, random_state=seed, stratify=df.y)
        tr, va = train_test_split(tr, test_size=0.25, random_state=seed, stratify=df.y.iloc[tr])
        D = {"tr": df.iloc[tr], "va": df.iloc[va], "te": df.iloc[te]}
        prep = Prep().fit(D["tr"])
        X = {s: prep.transform(D[s]) for s in D}   # (raw, tf)
        y = {s: D[s]["y"].values for s in D}
        yb = {s: (y[s] != 0).astype(int) for s in D}
        sp = {s: D[s]["static_final_state_check_pass"].values for s in D}
        names = prep.names()
        prior = np.bincount(y["tr"], minlength=5) / len(y["tr"])
        res_seed = {}

        # ---- static final-state check
        s_te = 1.0 - sp["te"]
        P5 = np.zeros((len(te), 5)); P5[np.arange(len(te)), np.where(sp["te"] == 1, 0, 1)] = 1
        m, yh = eval_all(y["te"], P5, s_te, 0.5, sp["te"], rng)
        res_seed["Static check"] = (m, s_te, yh)

        # ---- LR / DT baselines (raw and +TF)
        for tag, j in [("", 0), ("+TF", 1)]:
            best = None
            for C in [0.01, 0.1, 1.0, 10.0]:
                lr = LogisticRegression(C=C, max_iter=3000)
                lr.fit(X["tr"][j], y["tr"])
                f = f1_score(y["va"], lr.predict(X["va"][j]), average="macro")
                if best is None or f > best[0]:
                    best = (f, C, lr)
            _, C, lr = best
            Pva = lr.predict_proba(X["va"][j]); thr = best_threshold(yb["va"], 1 - Pva[:, 0]); a = tune_alpha(y["va"], Pva, prior)
            P5 = lr.predict_proba(X["te"][j]); s = 1 - P5[:, 0]
            m, yh = eval_all(y["te"], P5, s, thr, sp["te"], rng, prior, a); res_seed["LR" + tag] = (m, s, yh)
            hp_rows.append({"seed": seed, "model": "LR" + tag, "C": C, "alpha": a})
            best = None
            for depth in [4, 6, 8, 10, 12]:
                for leaf in [10, 30, 60]:
                    dt = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, random_state=seed)
                    dt.fit(X["tr"][j], y["tr"])
                    f = f1_score(y["va"], dt.predict(X["va"][j]), average="macro")
                    if best is None or f > best[0]:
                        best = (f, depth, leaf, dt)
            _, depth, leaf, dt = best
            Pva = dt.predict_proba(X["va"][j]); thr = best_threshold(yb["va"], 1 - Pva[:, 0]); a = tune_alpha(y["va"], Pva, prior)
            P5 = dt.predict_proba(X["te"][j]); s = 1 - P5[:, 0]
            m, yh = eval_all(y["te"], P5, s, thr, sp["te"], rng, prior, a); res_seed["DT" + tag] = (m, s, yh)
            hp_rows.append({"seed": seed, "model": "DT" + tag, "max_depth": depth, "min_samples_leaf": leaf, "alpha": a})

        # ---- proposed DW-kNN-TV and ablations
        def run_knn(Xtr, Xva, Xte, weighted_metric, p_fixed=None, a_fixed=None):
            w = mi_weights(Xtr, y["tr"], seed) if weighted_metric else np.ones(Xtr.shape[1])
            knn = DWKNN().fit(Xtr, y["tr"], w)
            dva, iva = knn.neighbours(Xva)
            grid = {}
            for k in K_GRID:
                for p in (P_GRID if p_fixed is None else [p_fixed]):
                    Pv = knn.proba(dva, iva, k, p)
                    for a in (A_GRID if a_fixed is None else [a_fixed]):
                        grid[(k, p, a)] = f1_score(y["va"], decide(Pv, prior, a), average="macro")
            (k, p, a) = max(grid, key=grid.get)
            Pv = knn.proba(dva, iva, k, p); thr = best_threshold(yb["va"], 1 - Pv[:, 0])
            dte, ite = knn.neighbours(Xte)
            P5 = knn.proba(dte, ite, k, p); s = 1 - P5[:, 0]
            m, yh = eval_all(y["te"], P5, s, thr, sp["te"], rng, prior, a)
            m["k"], m["p"], m["alpha"] = k, p, a
            return m, s, yh, grid, w, (knn, dte, ite, thr)

        m, s, yh, grid, w, aux = run_knn(X["tr"][1], X["va"][1], X["te"][1], True)
        res_seed["DW-kNN-TV"] = (m, s, yh)
        hp_rows.append({"seed": seed, "model": "DW-kNN-TV", "k": m["k"], "p": m["p"], "alpha": m["alpha"]})
        for (k, p, a), f in grid.items():
            if a == 0.5:
                sens_rows.append({"seed": seed, "k": k, "p": p, "val_macro_F1": f})
        mi_store.append(pd.Series(w, index=names))
        # test-set sensitivity (reported only as a check, not used for selection)
        knn, dte, ite, thr = aux
        for k in K_GRID:
            for p in P_GRID:
                Pt = knn.proba(dte, ite, k, p)
                sens_rows.append({"seed": seed, "k": k, "p": p, "test_macro_F1": f1_score(y["te"], decide(Pt, prior, 0.5), average="macro")})
        abl = {"Full DW-kNN-TV": m}
        abl["Uniform votes (p = 0)"] = run_knn(X["tr"][1], X["va"][1], X["te"][1], True, p_fixed=0.0)[0]
        abl["No prior correction (alpha = 0)"] = run_knn(X["tr"][1], X["va"][1], X["te"][1], True, a_fixed=0.0)[0]
        abl["No MI metric weighting"] = run_knn(X["tr"][1], X["va"][1], X["te"][1], False)[0]
        abl["No transition features"] = run_knn(X["tr"][0], X["va"][0], X["te"][0], True)[0]
        abl["Plain kNN (no TF, MI, p, alpha)"] = run_knn(X["tr"][0], X["va"][0], X["te"][0], False, p_fixed=0.0, a_fixed=0.0)[0]
        res_seed["Uniform kNN"] = (abl["Plain kNN (no TF, MI, p, alpha)"], None, None)
        for a, mm in abl.items():
            abl_rows.append({"seed": seed, "variant": a, **{kk: mm[kk] for kk in
                             ["F1_unsafe", "PR_AUC", "Macro_F1_5class", "Recall_static_pass_unsafe", "Recall@20%", "k", "p", "alpha"]}})

        for name, (mm, s, yh) in res_seed.items():
            rows.append({"seed": seed, "method": name, **{kk: v for kk, v in mm.items() if kk not in ("k", "p")}})
            if s is not None:
                for b in BUDGETS:
                    budget_rows.append({"seed": seed, "method": name, "budget": b,
                                        "recall": recall_at_budget(yb["te"], s, b, np.random.default_rng(seed))})
        if seed == 0:
            curves = {n: (yb["te"], res_seed[n][1]) for n in ["DW-kNN-TV", "LR+TF", "DT+TF", "LR", "DT", "Static check"]}
        cm_store[seed] = confusion_matrix(y["te"], res_seed["DW-kNN-TV"][2], labels=range(5))

        # ---- exposure-window regression (rows whose replay showed an exposure window)
        def regset(s):
            msk = D[s]["exposure_window_s"].values > 0
            return X[s][1][msk], np.log1p(D[s]["exposure_window_s"].values[msk])
        Xr_tr, yr_tr = regset("tr"); Xr_va, yr_va = regset("va"); Xr_te, yr_te = regset("te")
        wr = mi_weights(Xr_tr, np.digitize(yr_tr, np.quantile(yr_tr, [0.25, 0.5, 0.75])), seed)
        nn = NearestNeighbors(n_neighbors=max(K_GRID), n_jobs=1).fit(Xr_tr * np.sqrt(wr))
        dv, iv = nn.kneighbors(Xr_va * np.sqrt(wr)); dt_, it_ = nn.kneighbors(Xr_te * np.sqrt(wr))
        def kreg(d, i, k, p):
            ww = 1 / (d[:, :k] + EPS) ** p
            return (ww * yr_tr[i[:, :k]]).sum(1) / ww.sum(1)
        kb, pb = max(((k, p) for k in K_GRID for p in P_GRID), key=lambda kp: r2_score(yr_va, kreg(dv, iv, *kp)))
        reg_rows.append({"seed": seed, "method": "DW-kNN-TV", "R2_log": r2_score(yr_te, kreg(dt_, it_, kb, pb)), "n_test": len(yr_te)})
        rg = max((Ridge(alpha=a).fit(Xr_tr, yr_tr) for a in [0.1, 1, 10, 100]), key=lambda mdl: r2_score(yr_va, mdl.predict(Xr_va)))
        reg_rows.append({"seed": seed, "method": "Ridge (linear)", "R2_log": r2_score(yr_te, rg.predict(Xr_te)), "n_test": len(yr_te)})
        tr_ = max((DecisionTreeRegressor(max_depth=dd, min_samples_leaf=20, random_state=seed).fit(Xr_tr, yr_tr) for dd in [3, 5, 7, 9]),
                  key=lambda mdl: r2_score(yr_va, mdl.predict(Xr_va)))
        reg_rows.append({"seed": seed, "method": "DT regressor", "R2_log": r2_score(yr_te, tr_.predict(Xr_te)), "n_test": len(yr_te)})
        log(f"seed {seed}: kNN k={m['k']} p={m['p']} F1={m['F1_unsafe']:.3f} mF1={m['Macro_F1_5class']:.3f} | "
            f"LR+TF F1={res_seed['LR+TF'][0]['F1_unsafe']:.3f} DT+TF F1={res_seed['DT+TF'][0]['F1_unsafe']:.3f} "
            f"static F1={res_seed['Static check'][0]['F1_unsafe']:.3f}  ({time.time() - t0:.0f}s)")

    # ---------------------------------------------------------------- timing (seed-0 split, best settings)
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.2, random_state=0, stratify=df.y)
    prep = Prep().fit(df.iloc[tr]); Xtr = prep.transform(df.iloc[tr])[1]; Xte = prep.transform(df.iloc[te])[1]
    ytr = df.y.values[tr]
    trow = []
    for name, mk in [("LR+TF", lambda: LogisticRegression(C=1.0, max_iter=3000)),
                     ("DT+TF", lambda: DecisionTreeClassifier(max_depth=8, min_samples_leaf=30, random_state=0))]:
        t1 = time.perf_counter(); mdl = mk().fit(Xtr, ytr); t2 = time.perf_counter(); mdl.predict_proba(Xte); t3 = time.perf_counter()
        trow.append({"method": name, "fit_s": t2 - t1, "query_ms_per_patch": (t3 - t2) / len(te) * 1e3})
    t1 = time.perf_counter(); w = mi_weights(Xtr, ytr, 0); knn = DWKNN(k_max=40).fit(Xtr, ytr, w); t2 = time.perf_counter()
    d_, i_ = knn.neighbours(Xte); knn.proba(d_, i_, 40, 1.0); t3 = time.perf_counter()
    knn2 = DWKNN(k_max=40).fit(Xtr, ytr, w); t4 = time.perf_counter()
    trow.append({"method": "DW-kNN-TV", "fit_s": t2 - t1, "query_ms_per_patch": (t3 - t2) / len(te) * 1e3,
                 "index_rebuild_s_without_MI": t4 - t3})
    T = pd.DataFrame(trow); T.to_csv(os.path.join(RES, "timing.csv"), index=False); log(T.to_string())

    # ---------------------------------------------------------------- aggregate
    R = pd.DataFrame(rows); R.to_csv(os.path.join(RES, "main_per_seed.csv"), index=False)
    order = ["Static check", "LR", "DT", "LR+TF", "DT+TF", "Uniform kNN", "DW-kNN-TV"]
    keys = ["F1_unsafe", "Precision", "Recall", "PR_AUC", "ROC_AUC", "Macro_F1_5class", "Bal_acc_5class",
            "Recall_static_pass_unsafe", "FPR_safe", "Recall@20%"]
    agg = R.groupby("method")[keys].agg(["mean", "std"]).reindex(order)
    tab = pd.DataFrame({k: [f"{agg.loc[mth, (k, 'mean')]:.3f} ± {agg.loc[mth, (k, 'std')]:.3f}" for mth in order] for k in keys}, index=order)
    tab.index.name = "method"; tab.to_csv(os.path.join(RES, "table3_main_comparison.csv"))
    agg.to_csv(os.path.join(RES, "main_comparison_mean_std.csv"))
    det_keys = [f"detect_{c}" for c in CLASSES[1:]] + ["recall_safe"]
    R.groupby("method")[det_keys].agg(["mean", "std"]).reindex(order).to_csv(os.path.join(RES, "per_class_detection.csv"))

    A = pd.DataFrame(abl_rows); A.to_csv(os.path.join(RES, "ablation_per_seed.csv"), index=False)
    av = list(dict.fromkeys(A.variant))
    aagg = A.groupby("variant")[["F1_unsafe", "PR_AUC", "Macro_F1_5class", "Recall_static_pass_unsafe", "Recall@20%"]].agg(["mean", "std"]).reindex(av)
    aagg.to_csv(os.path.join(RES, "table4_ablation.csv"))

    S = pd.DataFrame(sens_rows)
    Sv = S.dropna(subset=["val_macro_F1"]).groupby(["k", "p"]).val_macro_F1.mean().unstack()
    St = S.dropna(subset=["test_macro_F1"]).groupby(["k", "p"]).test_macro_F1.mean().unstack()
    Sv.to_csv(os.path.join(RES, "sensitivity_k_p_val_macroF1.csv")); St.to_csv(os.path.join(RES, "sensitivity_k_p_test_macroF1.csv"))

    B = pd.DataFrame(budget_rows); B.to_csv(os.path.join(RES, "recall_at_budget_per_seed.csv"), index=False)
    Bagg = B.groupby(["method", "budget"]).recall.agg(["mean", "std"]).reset_index()
    Bagg.to_csv(os.path.join(RES, "recall_at_budget.csv"), index=False)

    G = pd.DataFrame(reg_rows); G.to_csv(os.path.join(RES, "exposure_regression_per_seed.csv"), index=False)
    Gagg = G.groupby("method").R2_log.agg(["mean", "std"]); Gagg.to_csv(os.path.join(RES, "exposure_regression.csv"))

    H = pd.DataFrame(hp_rows); H.to_csv(os.path.join(RES, "selected_hyperparameters.csv"), index=False)
    MI = pd.concat(mi_store, axis=1).mean(1).sort_values(ascending=False)
    MI.to_csv(os.path.join(RES, "mi_feature_weights.csv"), header=["mean_weight"])

    # paired t-tests vs baselines
    tt = []
    pv = R.pivot(index="seed", columns="method")
    for mtr in ["F1_unsafe", "PR_AUC", "Macro_F1_5class", "Recall@20%", "Recall_static_pass_unsafe"]:
        for b in ["Static check", "LR", "DT", "LR+TF", "DT+TF", "Uniform kNN"]:
            a, c = pv[(mtr, "DW-kNN-TV")].values, pv[(mtr, b)].values
            t, p = stats.ttest_rel(a, c)
            tt.append({"metric": mtr, "baseline": b, "mean_diff": (a - c).mean(), "t": t, "p_value": p})
    pd.DataFrame(tt).to_csv(os.path.join(RES, "paired_ttests.csv"), index=False)

    cm = sum(cm_store.values()); cmn = cm / cm.sum(1, keepdims=True)
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(os.path.join(RES, "confusion_matrix_dwknn_sum5seeds.csv"))

    # ---------------------------------------------------------------- figures
    def save(fig, name):
        fig.savefig(os.path.join(FIG, name + ".png"), dpi=300, bbox_inches="tight")
        fig.savefig(os.path.join(RES, name + ".jpg"), dpi=200, bbox_inches="tight", pil_kwargs={"quality": 92})
        plt.close(fig)

    # Fig 2: PR curves
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for n in ["DW-kNN-TV", "LR+TF", "DT+TF", "LR", "DT"]:
        yy, ss = curves[n]; pr, rc, _ = precision_recall_curve(yy, ss)
        ax.plot(rc, pr, color=COL[n], lw=1.8 if n == "DW-kNN-TV" else 1.1,
                ls="-" if "TF" in n or n == "DW-kNN-TV" else "--", label=f"{n} (AP={average_precision_score(yy, ss):.3f})")
    yy, ss = curves["Static check"]
    ax.scatter([recall_score(yy, ss >= .5)], [precision_score(yy, ss >= .5)], marker="s", color=COL["Static check"],
               zorder=5, label="Static check (point)")
    ax.axhline(yy.mean(), color="grey", ls=":", lw=0.8)
    ax.set_xlabel("Recall of unsafe patches"); ax.set_ylabel("Precision"); ax.set_xlim(0, 1); ax.set_ylim(0.25, 1.02)
    ax.legend(loc="lower left", frameon=False); ax.grid(alpha=0.3)
    save(fig, "fig2_pr_curves")

    # Fig 3: recall vs replay budget
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for n in ["DW-kNN-TV", "LR+TF", "DT+TF", "LR", "Static check"]:
        b = Bagg[Bagg.method == n]
        ax.errorbar(b.budget * 100, b["mean"], yerr=b["std"], color=COL[n], marker="o", ms=3, lw=1.6 if n == "DW-kNN-TV" else 1.0,
                    capsize=2, label=n)
    base = (df.y != 0).mean()
    xs = np.array(BUDGETS) * 100
    ax.plot(xs, np.minimum(xs / 100 / base, 1), color="k", ls=":", lw=0.8, label="Oracle")
    ax.set_xlabel("Sandbox-replay budget (% of patches)"); ax.set_ylabel("Unsafe patches caught (recall)")
    ax.legend(frameon=False, loc="lower right", ncol=2, fontsize=7, columnspacing=0.8); ax.grid(alpha=0.3); ax.set_ylim(0, 1.02)
    save(fig, "fig3_recall_budget")

    # Fig 4: sensitivity heatmap (validation macro-F1)
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    im = ax.imshow(Sv.values, cmap="Reds", aspect="auto")
    ax.set_xticks(range(len(Sv.columns))); ax.set_xticklabels([f"{c:g}" for c in Sv.columns])
    ax.set_yticks(range(len(Sv.index))); ax.set_yticklabels(Sv.index)
    ax.set_xlabel("Distance exponent p"); ax.set_ylabel("Neighbourhood size k")
    vmax = Sv.values.max()
    for i in range(Sv.shape[0]):
        for j in range(Sv.shape[1]):
            v = Sv.values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7,
                    color="white" if v > Sv.values.min() + 0.6 * (vmax - Sv.values.min()) else "black",
                    fontweight="bold" if v == vmax else "normal")
    cb = fig.colorbar(im, ax=ax); cb.set_label(r"Val. macro-F1 ($\alpha$ = 0.5)"); cb.ax.tick_params(labelsize=7)
    save(fig, "fig4_sensitivity_k_p")

    # Fig 5: confusion matrix
    fig, ax = plt.subplots(figsize=(3.3, 2.8))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(5)); ax.set_xticklabels(CLS_SHORT, rotation=35, ha="right")
    ax.set_yticks(range(5)); ax.set_yticklabels(CLS_SHORT)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=7.5, color="white" if cmn[i, j] > 0.55 else "black")
    ax.set_xlabel("Predicted outcome"); ax.set_ylabel("Replay outcome (true)")
    save(fig, "fig5_confusion")

    # Fig 6: per-hazard detection rates
    D6 = R.groupby("method")[det_keys].agg(["mean", "std"])
    meths = ["Static check", "LR", "DT+TF", "LR+TF", "DW-kNN-TV"]
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    wdt = 0.16; xs = np.arange(len(det_keys))
    for i, n in enumerate(meths):
        ax.bar(xs + (i - 2) * wdt, D6.loc[n].xs("mean", level=1), wdt, yerr=D6.loc[n].xs("std", level=1),
               color=COL[n], label=n, capsize=1.5, error_kw={"lw": 0.6})
    ax.set_xticks(xs); ax.set_xticklabels(["Final\ninsec.", "Trans.\nexp.", "Trans.\noutage", "Exp.+\noutage", "Safe\n(spec.)"])
    ax.set_ylabel("Detection rate / specificity"); ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, ncol=3, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 1.22)); ax.grid(axis="y", alpha=0.3)
    save(fig, "fig6_per_hazard")

    # Fig 7: MI feature weights (top 12)
    top = MI.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    ax.barh(range(len(top)), top.values, color=[COL["DW-kNN-TV"] if n in TF else "#999999" for n in top.index])
    ax.set_yticks(range(len(top))); ax.set_yticklabels([n.replace("_", " ") for n in top.index], fontsize=7)
    ax.set_xlabel("Metric weight $w_j$ (mean over 5 seeds)"); ax.grid(axis="x", alpha=0.3)
    save(fig, "fig7_metric_weights")

    # flowchart (Graphviz)
    dot = os.path.join(HERE, "fig1_flowchart.dot")
    if os.path.exists(dot) and shutil.which("dot"):
        subprocess.run(["dot", "-Tpng", "-Gdpi=300", dot, "-o", os.path.join(FIG, "fig1_flowchart.png")], check=True)
        subprocess.run(["dot", "-Tjpg", "-Gdpi=200", dot, "-o", os.path.join(RES, "fig1_flowchart.jpg")], check=True)

    metrics = {
        "main_mean": {m: {k: float(agg.loc[m, (k, "mean")]) for k in keys} for m in order},
        "main_std": {m: {k: float(agg.loc[m, (k, "std")]) for k in keys} for m in order},
        "ablation_mean": {v: {k: float(aagg.loc[v, (k, "mean")]) for k in ["F1_unsafe", "PR_AUC", "Macro_F1_5class", "Recall_static_pass_unsafe", "Recall@20%"]} for v in av},
        "exposure_regression_R2_log": {m: [float(Gagg.loc[m, "mean"]), float(Gagg.loc[m, "std"])] for m in Gagg.index},
        "selected_k_p_alpha": H[H.model == "DW-kNN-TV"][["seed", "k", "p", "alpha"]].to_dict("records"),
        "timing": trow,
        "runtime_s": time.time() - t0,
    }
    with open(os.path.join(RES, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    log(tab[["F1_unsafe", "PR_AUC", "Macro_F1_5class", "Recall_static_pass_unsafe", "Recall@20%"]].to_string())
    log(aagg.round(4).to_string()); log(Gagg.round(4).to_string())
    log(f"Total runtime {time.time() - t0:.1f}s")
    with open(os.path.join(RES, "run_log.txt"), "w") as fh:
        fh.write("\n".join(LOG))


if __name__ == "__main__":
    main()
