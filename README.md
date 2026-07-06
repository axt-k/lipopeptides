# Cstarter-domain substrate specificity prediction

Cstarters is a substrate specificity prediction pipeline for condenstation starter domains.

## Installation

Install dependencies:

```bash
conda create -n cstarters python=3.11
conda activate cstarters
git clone https://github.com/axt-k/lipopeptides.git
cd lipopeptides
python -m pip install --upgrade pip
pip install -e .
```

Install MAFFT dependency:

```bash
# TODO
```

## Use trained models for prediction

Run the help command to see all options:

```
conda activate cstarters
cstarters --help
```

## Cheminformatics analysis

The cheminformatics analysis is confined to a single script:

```
conda activate cstarters
python3 scripts/acylstarter_cheminformatics_analysis.py --data data/dataset.csv --out-dir out/
```