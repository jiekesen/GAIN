# GAIN
GAIN, a deep learning framework for predicting crop developmental potential from genotype and environmental data.

## System requirements
Python 3.10 / 3.11.

Optional: Hardware accelerator supporting PyTorch.
## Install GAIN

We provided a pre-packaged Conda environment for directly running Cropformer.

```bash
conda env create -f GAIN.yml
```
## Run GAIN
We provide a step-by-step guide for running GAIN.

## Train models using genotype data

```bash
python Run_GAIN_Genotype.py --vae_input ./data/geno_data.csv --label ./data/Pheotype.csv 
```
## Prediction based on genotype Data

```bash
python Geno_Predict.py ./data/geno_test.csv
```

## Training GAIN using genotype data and environmental data

```bash
python ./code_env/run_code.py ./data/env_geno_data.csv
```
## Prediction based on genotype and environmental data
```bash
python ./code_env/pre.py ./data/env_test_data.csv
```





