import pandas as pd 
from rdkit import Chem, RDLogger 
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")


def _largest_fragment(mol):
    frags = Chem.GetMolFrags(mol, asMols=True)
    return max(frags, key=lambda m: m.GetNumHeavyAtoms())


def load_clean(path="../data/bbbp.csv", verbose=True):
    df = pd.read_csv(path)
    n_start = len(df)
    
    df["mol"] = df["SMILES"].apply(Chem.MolFromSmiles)
    df = df.dropna(subset=["mol"]).copy() 
    df["mol"] = df["mol"].apply(_largest_fragment)
    df["clean_smiles"] = df["mol"].apply(Chem.MolToSmiles)
    
    # remove molecules whose duplicates disagree on the label 
    n_labels = df.groupby("clean_smiles")["label"].transform("nunique")
    n_conflict = int((n_labels > 1).sum())
    df = df[n_labels == 1]
    
    # remove remaining exact duplicates (same molecule, same label)
    n_before_dedup = len(df)
    df = df.drop_duplicates(subset="clean_smiles").copy()
    n_dups = n_before_dedup - len(df)

    df["scaffold"] = df["clean_smiles"].apply(
        lambda s: MurckoScaffold.MurckoScaffoldSmiles(smiles=s, includeChirality=False)
    )
    df = df.reset_index(drop=True)

    if verbose:
        print(f"Start: {n_start} | removed conflicting-label rows: {n_conflict} "
              f"| removed duplicates: {n_dups} | final: {len(df)}")
    return df