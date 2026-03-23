#!/usr/bin/env bash
set -e
conda env create -f environment.yml || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dat675-torch
python -m ipykernel install --user --name dat675-torch --display-name "Python (dat675-torch)"
jupyter notebook
