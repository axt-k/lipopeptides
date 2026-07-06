# !/usr/bin/env python3

import argparse
import subprocess
import tempfile
import sys
import csv

import pandas as pd

ALIGNMENT_PATH = "alignment.fasta"

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

def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", type=str, required=True)
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


def import_aa_properties(aa_properties_csv="16_aa_properties.csv"):
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
    X_pred = pd.DataFrame()

    for aa_property in positions:
        for pos in positions[aa_property]:
            X_pred[f"{aa_property}_{pos}"] = [aa_properties_ref[input_sequence[pos]][aa_property]]

    return X_pred


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

    predicted_sequence = args.fasta

    with tempfile.TemporaryDirectory() as temp_dir:
        output_alignment = f"{temp_dir}/output_alignment.fasta"

        temp_predicted_sequence_path = f"{temp_dir}/predicted_sequence.fasta"
        with open(temp_predicted_sequence_path, "w") as f:
            f.write(f">predicted_sequence\n{predicted_sequence}\n")
        run_mafft_add(ALIGNMENT_PATH, temp_predicted_sequence_path, output_alignment)

        parsed_fasta = parse_fasta(output_alignment)

        predicted_sequence_aligned = parsed_fasta["predicted_sequence"]

        aa_properties_ref = import_aa_properties("16_aa_properties.csv")

        X_aromaticity = featurize_alignment(predicted_sequence_aligned, PROPERTIES_AROMATICITY, aa_properties_ref)

        X_hydroxylation = featurize_alignment(predicted_sequence_aligned, PROPERTIES_HYDROXYLATION, aa_properties_ref)

        avg_L = average_value_from_seq(predicted_sequence_aligned, AVG_L_POSITIONS, "vdwvol", aa_properties_ref)
        avg_S = average_value_from_seq(predicted_sequence_aligned, AVG_S_POSITIONS, "vdwvol", aa_properties_ref)
        X_length = featurize_alignment(predicted_sequence_aligned, PROPERTIES_SIZE, aa_properties_ref)
        X_length.insert(0, "avg_size_tunnel_L", avg_L)
        X_length.insert(1, "avg_size_tunnel_S", avg_S)

        ml_aromaticity_model = unpickle_model("../../../src/cstarters/data/ml_aromaticity.pkl")
        ml_hydroxylation_model = unpickle_model("../../../src/cstarters/data/ml_hydroxylation.pkl")
        ml_length_model = unpickle_model("../../../src/cstarters/data/ml_length.pkl")

        aromaticity_prediction = ml_aromaticity_model.predict(X_aromaticity)
        hydroxylation_prediction = ml_hydroxylation_model.predict(X_hydroxylation)
        length_prediction = ml_length_model.predict(X_length)

        print(f"Predicted aromaticity: {aromaticity_prediction[0]}")
        print(f"Predicted hydroxylation: {hydroxylation_prediction[0]}")
        print(f"Predicted length category: {length_prediction[0]}")


if __name__ == "__main__":
    main()
