import numpy as np 
from rdkit.Chem import Descriptors, rdFingerprintGenerator 

_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def morgan_fp(mols):
    return np.array([_gen.GetFingerprintAsNumPy(m) for m in mols])

def descriptors(mols):
    fns = [Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA,
           Descriptors.NumHDonors, Descriptors.NumHAcceptors,
           Descriptors.NumRotatableBonds]
    return np.array([[f(m) for f in fns] for m in mols])

DESC_NAMES = ["MolWt", "LogP", "TPSA", "HDonors", "HAcceptors", "RotBonds"]