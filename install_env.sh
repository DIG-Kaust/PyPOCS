#!/bin/bash
# 
# Installer for pypocs
# 
# Run: ./install_env.sh
# 
# M. Ravasi, 16/04/2023

echo 'Creating pypocs environment'

# create conda env
conda env create -f environment.yml
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pypocs
conda env list
echo 'Created and activated environment:' $(which python)

# check cupy works as expected
echo 'Checking cupy, cusigna  l, and pylops version and running a command...'
python -c 'import cupy as cp; print(cp.__version__); print(cp.ones(10))'
python -c 'import cusignal; print(cusignal.__version__)'
python -c 'import pylops; print(pylops.__version__)'

echo 'Done!'

