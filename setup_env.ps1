conda env create -f environment.yml
conda activate dat675-torch
python -m ipykernel install --user --name dat675-torch --display-name "Python (dat675-torch)"
jupyter notebook
