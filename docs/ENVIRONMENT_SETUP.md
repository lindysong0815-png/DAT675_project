# Environment setup

## Recommended
Use the conda environment in `environment.yml`.

### Windows (PowerShell)
```powershell
conda env create -f environment.yml
conda activate dat675-torch
python -m ipykernel install --user --name dat675-torch --display-name "Python (dat675-torch)"
jupyter notebook
```

### macOS / Linux
```bash
conda env create -f environment.yml
conda activate dat675-torch
python -m ipykernel install --user --name dat675-torch --display-name "Python (dat675-torch)"
jupyter notebook
```

## If torch or torch-geometric fails on Windows
1. Install the latest Microsoft Visual C++ Redistributable.
2. Reinstall torch in the active environment.
3. Install the torch-geometric wheel matching your torch version.

## Minimal verification
Run this after activation:
```python
import torch
import pandas as pd
from rdkit import Chem
print(torch.__version__)
print(pd.__version__)
print(Chem.MolFromSmiles("CCO") is not None)
```

## Suggested notebook data path
If the notebook is stored inside `notebooks/`, a robust relative path is:

```python
from pathlib import Path
DATA_PATH = Path("../data/raw/tox21.csv")
```

This avoids hard-coded local paths and makes the repository easier to run on another machine.
