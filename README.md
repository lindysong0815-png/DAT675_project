# DAT675 Project
## When Does Multi-Task Learning Help or Hurt Toxicity Prediction?

**Team members**
- Song Qiuge
- Kevin Cheng

## Project overview
This project investigates when **multi-task learning (MTL)** improves toxicity prediction and when it causes **negative transfer** on the **Tox21** dataset.

We compare:
- classical fingerprint-based baselines: **Logistic Regression** and **Random Forest**
- graph-based models: **GCN-STL**, **GCN-MTL**, **GCN-NR-MTL**, and **GCN-SR-MTL**
- evaluation under **repeated scaffold-based splits** with **multiple random seeds**

The main metrics are:
- **AUPRC** (primary metric, due to class imbalance)
- **AUROC** (secondary metric)

## Key Result

- Random Forest remains a strong baseline under scaffold split  
- Multi-task learning provides limited average improvement  
- Negative transfer is observed for several endpoints  
- SR tasks benefit more from MTL than NR tasks  


## Repository structure
```text
DAT675_project/
├── README.md
├── requirements.txt
├── environment.yml
├── .gitignore
├── setup_env.ps1
├── setup_env.sh
├── notebooks/
│   └── G15_DAT675.ipynb
├── docs/
│   ├── G15_DAT675.html
│   └── ENVIRONMENT_SETUP.md
├── data/
│   ├── raw/
│   └── processed/
├── results/
│   ├── figures/
│   ├── tables/
│   └── exports/
└── src/
    ├── __init__.py
    ├── preprocessing.py
    ├── splitting.py
    ├── features.py
    ├── baselines.py
    ├── gnn_models.py
    ├── training.py
    ├── evaluation.py
    └── visualization.py
```

## Recommended environment setup
### Option 1: Conda (recommended)
```bash
conda env create -f environment.yml
conda activate dat675-torch
jupyter notebook
```

### Option 2: pip
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
jupyter notebook
```

## How to run
1. Put the raw Tox21 CSV file in `data/raw/`.
2. Open `notebooks/G15_DAT675.ipynb`.
3. Update the notebook data path if needed. A robust choice is:
   ```python
   from pathlib import Path
   DATA_PATH = Path("../data/raw/tox21.csv")
   ```
4. Run the notebook from top to bottom.
5. Export final figures to `results/figures/` and final tables to `results/tables/`.

## Main workflow
1. Data profiling and preprocessing  
2. Scaffold-based data splitting  
3. Feature engineering  
4. Baseline models  
5. GCN models  
6. Training pipeline  
7. Repeated evaluation and statistical testing  
8. Transfer analysis  
9. Visualization  
10. Final tables and reporting

## Notes
- The notebook is the **main deliverable**.
- The `src/` folder contains modular helper functions matching the notebook workflow.
- The project is structured for **clarity, reproducibility, and GitHub submission**.
- Published Tox21 benchmark values should be interpreted as **contextual references**, not a strict leaderboard, because split protocols differ across studies.
