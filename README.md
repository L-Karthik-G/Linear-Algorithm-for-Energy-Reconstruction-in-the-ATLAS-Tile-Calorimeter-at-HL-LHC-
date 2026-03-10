# ATLAS Tile Calorimeter — Linear Energy Reconstruction
### GSoC 2026 Evaluation Test | HSF ATLAS Project

A linear algorithm for reconstructing particle collision energy from digital detector samples in the ATLAS Tile Calorimeter at HL-LHC.

## Method
Ordinary Least Squares linear regression using a 7-sample sliding 
window (n-3 to n+3) to predict true energy at each bunch crossing.

## Requirements
pip install torch numpy scikit-learn matplotlib

## Usage
python linear_reg_model.py

> Note: the file path in the glob.glob() lines may need to be changes accordinginly
