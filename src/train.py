import os 
import numpy as np 
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

from src.data import load_clean 
from src.split import scaffold_split 
from src.features import morgan_fp, descriptors 

DATA = "data/bbbp.csv"


def ensure_data():
    if not os.path.exists(DATA):
        from datasets import load_dataset
        os.makedirs("data", exist_ok=True)
        ds = load_dataset("scikit-fingerprints/MoleculeNet_BBBP")["train"]
        ds.to_pandas().to_csv(DATA, index=False)


def make_rf():
    return RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                  random_state=42, n_jobs=-1)


def cv_auc(X, y, splitter, groups=None):
    aucs = []
    for tr_i, te_i in splitter.split(X, y, groups):
        m = make_rf().fit(X[tr_i], y[tr_i])
        aucs.append(roc_auc_score(y[te_i], m.predict_proba(X[te_i])[:, 1]))
    return np.mean(aucs), np.std(aucs)


def main():
    ensure_data()
    os.makedirs("results", exist_ok=True)

    df = load_clean(DATA)
    y = df["label"].values
    tr, va, te = scaffold_split(df)
    groups = df["scaffold"].astype("category").cat.codes.values

    X_fp = morgan_fp(df["mol"])
    X_desc = descriptors(df["mol"])
    feature_sets = {"fingerprint": X_fp, "descriptors": X_desc,
                    "both": np.hstack([X_fp, X_desc])}

    rows = []
    for name, X in feature_sets.items():
        m = make_rf().fit(X[tr], y[tr])
        p_va = m.predict_proba(X[va])[:, 1]
        p_te = m.predict_proba(X[te])[:, 1]
        ts = np.arange(0.3, 0.95, 0.05)
        t = max(ts, key=lambda t: balanced_accuracy_score(y[va], p_va >= t))
        pred = (p_te >= t).astype(int)

        r_mean, r_std = cv_auc(X, y, StratifiedKFold(5, shuffle=True, random_state=42))
        s_mean, s_std = cv_auc(X, y, StratifiedGroupKFold(5, shuffle=True, random_state=42), groups)

        rows.append({
            "features": name,
            "test_auc_fixed_scaffold": round(roc_auc_score(y[te], p_te), 3),
            "threshold_from_valid": round(float(t), 2),
            "test_recall_blocked": round(recall_score(y[te], pred, pos_label=0), 3),
            "test_recall_permeable": round(recall_score(y[te], pred, pos_label=1), 3),
            "cv_auc_random": f"{r_mean:.3f} +/- {r_std:.3f}",
            "cv_auc_scaffold": f"{s_mean:.3f} +/- {s_std:.3f}",
        })

    out = pd.DataFrame(rows)
    out.to_csv("results/metrics.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()