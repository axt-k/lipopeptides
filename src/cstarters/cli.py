# !/usr/bin/env python3

import argparse
import subprocess
import tempfile
import sys
import csv
import json
from importlib.resources import files
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError

import pandas as pd

import cstarters.data

PROPERTIES_AROMATICITY = {
    'WOLS870101': [936, 884, 924, 98],
    'WOLS870102': [350, 884],
    'WOLS870103': [98, 96, 818, 936, 986],
    'GRAR740102': [936, 98, 350],
    'RADA880108': [347],
    'NEU1': [936, 924]
}

PROPERTIES_HYDROXYLATION = {
    'WOLS870101': [924, 932, 944],
    'FAUJ880109': [991],
    'TSAJ990101': [814],
    'CHOP780203': [936, 818],
    'NEU1': [936],
    'NEU2': [98],
}

PROPERTIES_SIZE = {
    'WOLS870101': [96, 97, 98, 320, 338, 340, 342, 349, 351, 352, 356, 357, 814, 818, 902, 924, 932, 936, 938, 707, 602],
    'WOLS870102': [96, 97, 98, 320, 338, 340, 342, 349, 351, 352, 356, 357, 814, 818, 902, 924, 932, 936, 938, 707, 602],
    'WOLS870103': [96, 97, 98, 320, 338, 340, 342, 349, 351, 352, 356, 357, 814, 818, 902, 924, 932, 936, 938, 707, 602],
    'GRAR740102': [96, 97, 98, 320, 338, 340, 342, 349, 351, 352, 356, 357, 814, 818, 902, 924, 932, 936, 938, 707, 602],
    'RADA880108': [96, 97, 98, 320, 338, 340, 342, 349, 351, 352, 356, 357, 814, 818, 902, 924, 932, 936, 938, 707, 602]
}

AVG_L_POSITIONS = [96, 97, 98, 320, 338, 340, 342, 349, 351, 352, 356, 357, 814, 818, 902, 924, 932, 936, 938]
AVG_S_POSITIONS = [96, 97, 98, 349, 352, 814, 818, 924, 932, 936, 938, 944]

try:
    __version__ = version("cstarters")
except PackageNotFoundError:
    __version__ = "0.0.0"

def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--seq", type=str, required=True, help="path to file with input condensation-starter amino acid sequence")
    parser.add_argument("--out", type=str, required=False, default=None, help="optional path to output JSONL with predictions")
    return parser.parse_args()

def run_mafft_add(reference_alignment, predicted_sequence, output_alignment):
    mafft_command = [
        "mafft",
        "--add",
        predicted_sequence,
        "--keeplength",
        reference_alignment
    ]
    with open(output_alignment, "w") as out:
        subprocess.run(
            mafft_command,
            stdout=out,
            stderr=subprocess.DEVNULL,
            check=True,
        )


def parse_fasta(fasta_file):
    sequences = {}
    with open(fasta_file, "r") as f:
        current_header = None
        current_sequence = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_header is not None:
                    sequences[current_header] = "".join(current_sequence)
                current_header = line[1:]
                current_sequence = []
            else:
                current_sequence.append(line)
        if current_header is not None:
            sequences[current_header] = "".join(current_sequence)
    return sequences


def import_aa_properties():
    aa_properties_csv = Path(files(cstarters.data)).joinpath("16_aa_properties.csv")

    aa_properties_ref = {}
    with open(aa_properties_csv, 'r') as aa_properties_file:
        reader = csv.DictReader(aa_properties_file, delimiter=',')
        for row in reader:
            aa_properties_ref[row["AA_ABREV"]] = {}
            for aa_property in row:
                if aa_property != "AA_ABREV":
                    aa_properties_ref[row["AA_ABREV"]][aa_property] = float(row[aa_property])
    return aa_properties_ref

def featurize_alignment(input_sequence, positions, aa_properties_ref):
    features = {}

    for aa_property, aa_positions in positions.items():
        for pos in aa_positions:
            aa = input_sequence[pos]
            features[f"{aa_property}_{pos}"] = [aa_properties_ref[aa][aa_property]]

    return pd.DataFrame(features)

def average_value_from_seq(seq, positions, aa_property, aa_properties_ref):
    value_sum = 0.0
    aa_num = 0

    for pos in positions:
        aa = seq[int(pos)]

        if aa == "-":
            continue

        value_sum += aa_properties_ref[aa][aa_property]
        aa_num += 1

    if aa_num == 0:
        return float("nan")

    return value_sum / aa_num


def unpickle_model(model_path):
    import pickle
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def main() -> None:
    args = cli()

    with open(args.fasta, "r") as f:
        predicted_sequence = f.read()
    input_file_name_stem = Path(args.fasta).stem

    with tempfile.TemporaryDirectory() as temp_dir:
        output_alignment = f"{temp_dir}/output_alignment.fasta"

        temp_predicted_sequence_path = f"{temp_dir}/predicted_sequence.fasta"
        with open(temp_predicted_sequence_path, "w") as f:
            f.write(f">predicted_sequence\n{predicted_sequence}\n")

        ref_alignment = Path(files(cstarters.data)).joinpath("alignment.fasta")
        run_mafft_add(ref_alignment, temp_predicted_sequence_path, output_alignment)

        parsed_fasta = parse_fasta(output_alignment)

        predicted_sequence_aligned = parsed_fasta["predicted_sequence"]

        aa_properties_ref = import_aa_properties()

        X_aromaticity = featurize_alignment(predicted_sequence_aligned, PROPERTIES_AROMATICITY, aa_properties_ref)

        X_hydroxylation = featurize_alignment(predicted_sequence_aligned, PROPERTIES_HYDROXYLATION, aa_properties_ref)

        avg_L = average_value_from_seq(predicted_sequence_aligned, AVG_L_POSITIONS, "vdwvol", aa_properties_ref)
        avg_S = average_value_from_seq(predicted_sequence_aligned, AVG_S_POSITIONS, "vdwvol", aa_properties_ref)

        # X_length = featurize_alignment(predicted_sequence_aligned, PROPERTIES_SIZE, aa_properties_ref)
        # X_length.insert(0, "avg_size_tunnel_L", avg_L)
        # X_length.insert(1, "avg_size_tunnel_S", avg_S)
        X_length_features = featurize_alignment(predicted_sequence_aligned, PROPERTIES_SIZE, aa_properties_ref)
        X_length = pd.concat([pd.DataFrame({"avg_size_tunnel_L": [avg_L], "avg_size_tunnel_S": [avg_S]}), X_length_features], axis=1)

        ml_aromaticity_model_path = Path(files(cstarters.data).joinpath("ml_aromaticity.pkl"))
        ml_hydroxylation_model_path = Path(files(cstarters.data).joinpath("ml_hydroxylation.pkl"))
        ml_length_model_path = Path(files(cstarters.data).joinpath("ml_length.pkl"))

        ml_aromaticity_model = unpickle_model(ml_aromaticity_model_path)
        ml_hydroxylation_model = unpickle_model(ml_hydroxylation_model_path)
        ml_length_model = unpickle_model(ml_length_model_path)

        aromaticity_prediction = ml_aromaticity_model.predict(X_aromaticity)
        hydroxylation_prediction = ml_hydroxylation_model.predict(X_hydroxylation)
        length_prediction = ml_length_model.predict(X_length)

        result_dict = {
            "input_file_name": input_file_name_stem,
            "input_sequence": predicted_sequence,
            "type": str(aromaticity_prediction[0]),
            "hydroxylation": bool(hydroxylation_prediction[0]),
            "length": str(length_prediction[0][0]),
        }

        print(json.dumps(result_dict, indent=2))

        if args.out:
            with open(args.out, "w") as f_o:
                f_o.write(json.dumps(result_dict))


if __name__ == "__main__":
    main()
