"""Utilities for Tox21 preprocessing and profiling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import MolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import matthews_corrcoef

RDLogger.DisableLog("rdApp.*")


@dataclass(frozen=True)
class EndpointGroups:
    all_endpoints: list[str]
    nr_endpoints: list[str]
    sr_endpoints: list[str]


def build_default_endpoint_groups() -> EndpointGroups:
    all_endpoints = [
        "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
        "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
    ]
    nr_endpoints = [e for e in all_endpoints if e.startswith("NR")]
    sr_endpoints = [e for e in all_endpoints if e.startswith("SR")]
    return EndpointGroups(all_endpoints, nr_endpoints, sr_endpoints)


_normalizer = MolStandardize.rdMolStandardize.Normalizer()
_reionizer = MolStandardize.rdMolStandardize.Reionizer()
_uncharger = MolStandardize.rdMolStandardize.Uncharger()
_fragment_chooser = MolStandardize.rdMolStandardize.LargestFragmentChooser()


def standardize_smiles(smiles: str | None) -> Optional[str]:
    if pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = _fragment_chooser.choose(mol)
        mol = _normalizer.normalize(mol)
        mol = _reionizer.reionize(mol)
        mol = _uncharger.uncharge(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def clean_tox21_dataframe(df: pd.DataFrame, smiles_col: str = "smiles") -> pd.DataFrame:
    out = df.reset_index(drop=False).rename(columns={"index": "orig_idx"}).copy()
    out["CanonicalSMILES"] = out[smiles_col].apply(standardize_smiles)
    out = out.loc[out["CanonicalSMILES"].notna()].copy()
    out = out.drop_duplicates(subset="CanonicalSMILES").reset_index(drop=True)
    return out


def compute_missing_label_table(df: pd.DataFrame, endpoints: Iterable[str]) -> pd.DataFrame:
    n_total = len(df)
    rows = []
    for endpoint in endpoints:
        n_missing = int(df[endpoint].isna().sum())
        n_observed = n_total - n_missing
        rows.append({
            "Endpoint": endpoint,
            "Observed labels": n_observed,
            "Missing labels": n_missing,
            "Coverage": n_observed / n_total if n_total else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["Coverage", "Endpoint"]).reset_index(drop=True)


def compute_imbalance_table(df: pd.DataFrame, endpoints: Iterable[str]) -> pd.DataFrame:
    rows = []
    for endpoint in endpoints:
        y = df[endpoint].dropna()
        n = len(y)
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        rows.append({
            "Endpoint": endpoint,
            "Observed labels": n,
            "Positives": n_pos,
            "Negatives": n_neg,
            "Positive rate": n_pos / n if n else np.nan,
            "Neg/Pos ratio": n_neg / n_pos if n_pos else np.nan,
        })
    return pd.DataFrame(rows).sort_values("Positive rate").reset_index(drop=True)


def add_scaffolds(df: pd.DataFrame, smiles_col: str = "CanonicalSMILES") -> pd.DataFrame:
    out = df.copy()

    def get_scaffold(smiles: str | None) -> Optional[str]:
        if pd.isna(smiles):
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
            return scaffold or None
        except Exception:
            return None

    out["MurckoScaffold"] = out[smiles_col].apply(get_scaffold)
    out["EmptyScaffold"] = out["MurckoScaffold"].isna()
    out["Scaffold"] = out["MurckoScaffold"].fillna(out[smiles_col])
    return out


def compute_phi_matrix(df: pd.DataFrame, endpoints: Iterable[str]) -> pd.DataFrame:
    endpoints = list(endpoints)
    phi = pd.DataFrame(index=endpoints, columns=endpoints, dtype=float)
    for e1 in endpoints:
        for e2 in endpoints:
            mask = df[e1].notna() & df[e2].notna()
            y1 = df.loc[mask, e1]
            y2 = df.loc[mask, e2]
            if len(y1) == 0 or y1.nunique() < 2 or y2.nunique() < 2:
                phi.loc[e1, e2] = np.nan
            else:
                phi.loc[e1, e2] = matthews_corrcoef(y1, y2)
    return phi
