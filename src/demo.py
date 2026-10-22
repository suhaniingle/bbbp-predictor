import os 
import sys 

import joblib 
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw
from sklearn.ensemble import RandomForestClassifier

from src.data import load_clean, _largest_fragment
from src.features import morgan_fp, descriptors
from src.train import ensure_data

RDLogger.DisableLog("rdApp.*")
MODEL_PATH = "models/rf_both.joblib"


def get_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    ensure_data()
    df = load_clean("data/bbbp.csv", verbose=False)
    X = np.hstack([morgan_fp(df["mol"]), descriptors(df["mol"])])
    model = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                   random_state=42, n_jobs=-1)
    model.fit(X, df["label"].values)
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return model


def explain(smiles, out="results/demo.png"):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        sys.exit(f"Could not parse SMILES: {smiles}")
    mol = _largest_fragment(mol)
    train_smiles = set(load_clean("data/bbbp.csv", verbose=False)["clean_smiles"])
    seen = Chem.MolToSmiles(mol) in train_smiles

    model = get_model()
    desc = descriptors([mol])[0]
    x = np.hstack([morgan_fp([mol]), desc.reshape(1, -1)])
    p = float(model.predict_proba(x)[0, 1])
    mw, logp, tpsa, hbd, hba, rot = desc

    good = p >= 0.5
    color = "#2a9d8f" if good else "#e76f51"
    verdict = "likely to cross" if good else "likely blocked"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.2),
                                   gridspec_kw={"width_ratios": [1, 1.15]})

    ax1.imshow(Draw.MolToImage(mol, size=(500, 500)))
    ax1.axis("off")
    label = smiles if len(smiles) < 45 else smiles[:42] + "..."
    ax1.set_title(label, fontsize=9)

    ax2.axis("off")
    ax2.text(0, 0.95, "Blood-brain barrier prediction", fontsize=15,
             fontweight="bold", transform=ax2.transAxes)
    ax2.text(0, 0.86, f"{p:.0%} probability permeable ({verdict})",
             fontsize=12, color=color, fontweight="bold", transform=ax2.transAxes)

    bar = ax2.inset_axes([0, 0.72, 1, 0.09])
    bar.barh([0], [p], color=color)
    bar.barh([0], [1 - p], left=[p], color="#e5e5e5")
    bar.set_xlim(0, 1)
    bar.axis("off")

    ax2.text(0, 0.62, "Properties vs rough CNS-drug guidelines",
             fontsize=11, fontweight="bold", transform=ax2.transAxes)
    rows = [
        ("TPSA", f"{tpsa:.0f} A2", "guideline: <= 90", tpsa <= 90),
        ("Mol. weight", f"{mw:.0f}", "guideline: <= 450", mw <= 450),
        ("H-bond donors", f"{hbd:.0f}", "guideline: <= 3", hbd <= 3),
        ("LogP", f"{logp:.1f}", "", None),
    ]
    y = 0.53
    for name, val, rule, ok in rows:
        c = "#333333" if ok is None else ("#2a9d8f" if ok else "#e76f51")
        ax2.text(0, y, name, fontsize=11, transform=ax2.transAxes)
        ax2.text(0.35, y, val, fontsize=11, color=c, fontweight="bold",
                 transform=ax2.transAxes)
        ax2.text(0.52, y, rule, fontsize=10, color="#777777",
                 transform=ax2.transAxes)
        y -= 0.08

        if seen:
            ax2.text(0, 0.16, "In the training data: not an independent prediction",
                 fontsize=10, color="#e76f51", fontweight="bold",
                 transform=ax2.transAxes)
    else:
        ax2.text(0, 0.16, "Not in the training data: unseen molecule",
                 fontsize=10, color="#2a9d8f", fontweight="bold",
                 transform=ax2.transAxes)
        
    ax2.text(0, 0.02,
             "Random Forest on Morgan fingerprints + 6 descriptors, trained on 1,955 BBBP molecules.\n"
             "Research demo only: the model misses many non-penetrant molecules\n"
             "(blocked-class recall ~0.4 on the scaffold test set).",
             fontsize=8, color="#777777", transform=ax2.transAxes)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=200)
    print(f"{smiles} -> {p:.3f} ({verdict}); saved {out}")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Usage: python -m src.demo "<SMILES>" [output.png]')
    explain(sys.argv[1], *(sys.argv[2:3]))