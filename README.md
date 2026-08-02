# AT1_AML

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

AT1_ML

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         AT1_AML and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── AT1_AML   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes AT1_AML a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

## Reproducibility Instructions

### 1. Install dependencies and local package
To set up the project environment and install the required libraries:
```bash
pip install -r requirements.txt
```
This will automatically install our custom `nba_prep` feature engineering package in editable mode.

### 2. Preprocess Data and Engineer Features
Run the dataset pipeline to clean the raw data (height, year, age) and generate the advanced basketball features:
```bash
python -m AT1_AML.dataset
```
This will save the processed training and test features to `data/processed/train_processed.csv` and `data/processed/test_processed.csv`.

### 3. Train Model and Generate Predictions
Run the cross-validated model training script:
```bash
python -m AT1_AML.modeling.train
```
This will:
- Execute a 5-Fold Stratified Cross-Validation using LightGBM.
- Output overall AUPRC scores (target metric) and feature importances.
- Save the final model to `models/lgb_model.txt`.
- Save the predicted probabilities for the test set to `submission.csv` (ready for Kaggle upload).

### 4. Interactive Analysis
An interactive end-to-end walkthrough notebook is also available at `notebooks/36120-26SP-group-student-AT1-experiment-1.ipynb`.
