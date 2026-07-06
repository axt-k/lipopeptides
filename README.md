# Cstarter-domain substrate specificity prediction

Cstarters is a substrate specificity prediction pipeline for condenstation starter domains.

## Making predictions

### Installation

Create environment (see below for Apple Silicon):

```bash
conda create -n cstarters python=3.11
conda activate cstarters
```

On Apple Silicon, MAFFT is not available for the native ARM conda platform in some channels, so you can create an x86_64/Rosetta conda environment instead:

```bash
CONDA_SUBDIR=osx-64 conda create -n cstarters python=3.11
conda activate cstarters
conda config --env --set subdir osx-64
```

Install CLI with dependencies:

```bash
git clone https://github.com/axt-k/lipopeptides.git
cd lipopeptides
python -m pip install --upgrade pip
pip install -e .
conda install mafft==7.525
```

### Usage

Predict type (acyl or aromatic), beta-hydroxylation, and length category for an input condensation-starter amino acid sequence:

```bash
conda activate cstarters
cstarters --seq data/test/cstarter_aa_seq_BGC0000336.txt --out out/prediction.jsonl
```

```json
{
  "input_file_name": "cstarter_aa_seq_BGC0000336",
  "input_sequence": ...,
  "type": "FA",
  "hydroxylation": "FALSE",
  "length": "LCFA"
}
```

Run the help command to see all options:

```
conda activate cstarters
cstarters --help
```

## Analyses and training models

### Cheminformatics analysis

The cheminformatics analysis is confined to a single script:

```
conda activate cstarters
python3 scripts/acylstarter_cheminformatics_analysis.py --data data/dataset.csv --out-dir out/
```

### Train models

```bash
# TODO
```