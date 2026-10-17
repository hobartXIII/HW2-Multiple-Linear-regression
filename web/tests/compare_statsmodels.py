"""Check web/js/mlr.js against statsmodels on the same split, encoding and features.

usage (from HW2/web):  python tests/compare_statsmodels.py
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera

sys.stdout.reconfigure(encoding="utf-8")
WEB = Path(__file__).resolve().parents[1]
CASES = [
    ("data/flood_s4e5_sample.csv", "FloodProbability", "rfe", "20"),
    ("data/student_performance_factors.csv", "Exam_Score", "rfe", "auto"),
    ("data/student_performance_factors.csv", "Exam_Score", "forward", "8"),
    ("tests/50_Startups.csv", "Profit", "auto", "auto"),
]


def design(df, js):
    """Rebuild the JS design matrix: train medians/modes, sorted levels, drop first level."""
    tr = df.iloc[js["trainIdx"]]
    X = pd.DataFrame(index=df.index)
    for c in js["schema"]:
        name = c["name"]
        if c["excluded"] or name == TARGET:
            continue
        if c["kind"] == "numeric":
            col = pd.to_numeric(df[name], errors="coerce")
            X[name] = col.fillna(pd.to_numeric(tr[name], errors="coerce").median())
        else:
            col = df[name].astype("string").str.strip()
            col = col.fillna(tr[name].astype("string").str.strip().mode()[0])
            for lv in sorted(col.iloc[js["trainIdx"]].dropna().unique())[1:]:
                X[f"{name} = {lv}"] = (col == lv).astype(float)
    return X[js["selected"]]


def check(label, a, b, tol):
    err = float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float)) / np.maximum(1, np.abs(np.asarray(b, float)))))
    status = "OK " if err < tol else "BAD"
    print(f"  {status} {label:<28} max rel err {err:.2e}")
    return err < tol


ok = True
for csv, TARGET, method, k in CASES:
    out = WEB / "tests" / "_out.json"
    subprocess.run(["node", str(WEB / "tests" / "run_mlr.js"), str(WEB / csv), TARGET, method, k, str(out)], check=True)
    js = json.loads(out.read_text(encoding="utf-8"))
    df = pd.read_csv(WEB / csv, keep_default_na=True)
    X = design(df, js)
    y = pd.to_numeric(df[TARGET])
    Xtr, ytr = sm.add_constant(X.iloc[js["trainIdx"]], has_constant="add"), y.iloc[js["trainIdx"]]
    Xte, yte = sm.add_constant(X.iloc[js["testIdx"]], has_constant="add"), y.iloc[js["testIdx"]]
    m = sm.OLS(ytr, Xtr).fit()
    sf = m.get_prediction(Xte.iloc[:50]).summary_frame(alpha=0.05)
    pred = m.predict(Xte)
    r2 = 1 - ((yte - pred) ** 2).sum() / ((yte - yte.mean()) ** 2).sum()
    jc = pd.DataFrame(js["coefs"])
    tp = pd.DataFrame(js["testPred"])
    d = js["diag"]
    vif = [variance_inflation_factor(Xtr.values, i) for i in range(1, Xtr.shape[1])] if Xtr.shape[1] > 2 else [1.0]
    print(f"{csv} [{method}, k={k}] -> {len(js['selected'])} features")
    ok &= check("coefficients", jc.coef, m.params.values, 1e-6)
    ok &= check("std errors", jc.se, m.bse.values, 1e-6)
    ok &= check("p-values", jc.p, m.pvalues.values, 1e-6)
    ok &= check("coef CI", jc[["lo", "hi"]].values, m.conf_int().values, 1e-6)
    ok &= check("prediction", tp.pred, sf["mean"], 1e-6)
    ok &= check("confidence interval", tp[["ciLo", "ciHi"]].values, sf[["mean_ci_lower", "mean_ci_upper"]].values, 1e-6)
    ok &= check("prediction interval", tp[["piLo", "piHi"]].values, sf[["obs_ci_lower", "obs_ci_upper"]].values, 1e-6)
    ok &= check("test R2", js["metrics"]["R2"], r2, 1e-6)
    ok &= check("Durbin-Watson", d["dw"], durbin_watson(m.resid), 1e-6)
    ok &= check("Jarque-Bera", [d["jb"], d["jbP"]], jarque_bera(m.resid)[:2], 1e-6)
    ok &= check("Breusch-Pagan", [d["bp"], d["bpP"]], het_breuschpagan(m.resid, Xtr)[:2], 1e-5)
    ok &= check("VIF", d["vif"], vif, 1e-5)
print("ALL OK" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
