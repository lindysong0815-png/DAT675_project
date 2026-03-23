"""Fingerprint and graph feature helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def smiles_to_morgan_fp(smiles: str | None, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    arr = np.zeros((n_bits,), dtype=np.float32)
    if pd.isna(smiles):
        return arr
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return arr
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def add_morgan_fingerprints(df: pd.DataFrame, smiles_col: str = "CanonicalSMILES", fp_col: str = "MorganFP") -> pd.DataFrame:
    out = df.copy()
    out[fp_col] = out[smiles_col].apply(smiles_to_morgan_fp)
    return out


def get_atom_feature_dim() -> int:
    """Return the atom feature dimension used in the notebook implementation."""
    return 12 + 6 + 5 + 5 + 5 + 1 + 1


def get_bond_feature_dim() -> int:
    """Return the bond feature dimension used in the notebook implementation."""
    return 4 + 1 + 1 + 5
