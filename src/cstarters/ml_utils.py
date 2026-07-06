import csv

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.utils import shuffle


def avg(l):
    t = 0
    for i in l:
        t += i
    return t/len(l)

def load_dataset(path="cstarters_dataset.csv"):
    """Load the dataset from a CSV file."""
    dataset_df = pd.read_csv(path)
    relevant_columns = ["cs_accession", "starter_beta_hydroxylation", "starter_category", "aligned_sequence", "ncbi_organism_genus"]
    dataset_df = dataset_df[relevant_columns]
    
    dataset_df["starter_category"] = dataset_df["starter_category"].replace(
        ["unknown", "_not_elligible"],
        np.nan
    )
    missing_alignment = dataset_df[
        dataset_df["starter_category"].notna() &
        dataset_df["aligned_sequence"].isna()
    ]

    # print(missing_alignment)
    dataset_df = dataset_df.dropna(subset=["aligned_sequence"])
    dataset_df['aligned_sequence'] = dataset_df['aligned_sequence'].apply(lambda x:x.upper())

    # find duplicate cs_accessions and remove them. If duplicates differ in category, priority is as follows: AR > LCFA > MCFA > SCFA > FA
    # for beta hydroxylation, if duplicates differ, TRUE is prioritized over FALSE
    category_priority = {
        "AR": 0,
        "LCFA": 1,
        "MCFA": 2,
        "SCFA": 3,
        "FA": 4,
    }

    dataset_df["starter_category_priority"] = dataset_df["starter_category"].map(category_priority)
    dataset_df["starter_beta_hydroxylation_priority"] = dataset_df["starter_beta_hydroxylation"].astype(bool).astype(int)

    dataset_df = dataset_df.sort_values(by=["cs_accession", "starter_category_priority", "starter_beta_hydroxylation_priority"], ascending=[True, True, False])

    dataset_df = dataset_df.drop_duplicates(subset=["cs_accession"], keep="first")

    dataset_df = dataset_df.drop(columns=["starter_category_priority", "starter_beta_hydroxylation_priority"])
    return dataset_df

def add_stratify_column(dataset_df, column_names, predicted_column=None, min_count=2):
    dataset_df = dataset_df.copy()

    dataset_df["stratify_column"] = (
        dataset_df[column_names]
        .fillna("None")
        .astype(str)
        .agg("_".join, axis=1)
    )

    counts = dataset_df["stratify_column"].value_counts()
    rare_mask = dataset_df["stratify_column"].map(counts) < min_count

    if predicted_column is not None:
        dataset_df.loc[rare_mask, "stratify_column"] = (
            dataset_df.loc[rare_mask, predicted_column]
            .fillna("None")
            .astype(str)
        )
    else:
        dataset_df.loc[rare_mask, "stratify_column"] = "RARE"

    return dataset_df

def drop_labels(df, column, labels):
    """Remove rows where df[column] is in labels."""
    return df[~df[column].isin(labels)].copy()

def load_aa_properties(aa_properties_csv="16_aa_properties.csv"):
    aa_properties_ref = {}

    with open(aa_properties_csv, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            aa = row["AA_ABREV"]
            aa_properties_ref[aa] = {
                key: float(value)
                for key, value in row.items()
                if key != "AA_ABREV"
            }
    # print(aa_properties_ref)
    return aa_properties_ref


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


def average_value_feature(
    dataset_df,
    positions,
    aa_property,
    output_column,
    sequence_column="aligned_sequence",
    aa_properties_csv="16_aa_properties.csv",
    drop_sequence=False,
):
    aa_properties_ref = load_aa_properties(aa_properties_csv)

    ml_df = dataset_df.copy()

    ml_df[output_column] = ml_df[sequence_column].apply(
        lambda seq: average_value_from_seq(
            seq=seq,
            positions=positions,
            aa_property=aa_property,
            aa_properties_ref=aa_properties_ref,
        )
    )

    if drop_sequence:
        ml_df = ml_df.drop(columns=[sequence_column])

    return ml_df


def featurize_alignment(dataset_df, positions, aa_properties_csv="16_aa_properties.csv"):
    """Turn the aligned sequences into a feature matrix based on the amino acid properties at the specified positions."""
    aa_properties_ref = {}
    properties_list = []
    with open(aa_properties_csv, 'r') as aa_properties_file:
        reader = csv.DictReader(aa_properties_file, delimiter=',')
        for row in reader:
            aa_properties_ref[row["AA_ABREV"]] = {}
            for aa_property in row:
                if aa_property != "AA_ABREV":
                    aa_properties_ref[row["AA_ABREV"]][aa_property] = float(row[aa_property])
                    if not aa_property in properties_list:
                        properties_list.append(aa_property)
    ml_df = dataset_df.copy()
    for aa_property in positions:
        for pos in positions[aa_property]:
            ml_df[aa_property + '_' + str(pos)] = dataset_df['aligned_sequence'].apply(lambda x:aa_properties_ref[x[int(pos)]][aa_property])
            ml_df = ml_df.copy()
    
    return ml_df.drop(columns=["aligned_sequence"])

def ohe_genus(dataset_df, genus_column_name, threshold=5):
    dataset_df = dataset_df.copy()

    counts = dataset_df[genus_column_name].value_counts(dropna=True)

    common_genera = counts[counts >= threshold].index

    ohe = pd.get_dummies(dataset_df[genus_column_name])
    ohe = ohe.loc[:, common_genera]

    ohe = ohe.astype(int)

    ohe = ohe.add_prefix("genus_")

    dataset_df = pd.concat([dataset_df, ohe], axis=1)

    return dataset_df



def confusion_matrix_plot(cm, labels, figsize=(4, 3), cmap='Blues', title="Confusion Matrix"):
    """Plot a confusion matrix for the cross-validated predictions."""
    correct = 0
    total = 0
    for i in range(len(cm)):
        correct += cm[i][i]
        total += sum(cm[i])
    accuracy = correct / total       

    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title + f" (Accuracy: {accuracy:.3f})")
    plt.tight_layout()
    plt.show()


def plot_feature_importances(model, feature_names, combine_at_position=True, top_n=10):
    importances = pd.Series(
        model.feature_importances_,
        index=feature_names
    )

    if combine_at_position:
        positions = importances.index.to_series().str.split("_", n=1).str[1]

        importances = (
            importances
            .groupby(positions)
            .sum()
            .sort_values(ascending=False)
        )
    else:
        importances = importances.sort_values(ascending=False)

    top_importances = importances.head(top_n).sort_values()

    plt.figure(figsize=(8, max(4, top_n * 0.3)))
    top_importances.plot(kind="barh")
    plt.xlabel("Feature importance")
    plt.ylabel("Residue position" if combine_at_position else "Feature")
    plt.title(f"Top {top_n} feature importances")
    plt.tight_layout()
    plt.show()

    return importances


def plot_permutation_importances(
    model,
    X,
    y,
    combine_at_position=True,
    top_n=20,
    scoring="balanced_accuracy",
    n_repeats=30,
    random_state=42,
):
    result = permutation_importance(
        model,
        X,
        y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    importances = pd.Series(
        result.importances_mean,
        index=X.columns,
    )

    if combine_at_position:
        groups = [
            name.split("_", 1)[1]
            if "_" in name and name.split("_", 1)[1].isdigit()
            else name
            for name in importances.index
        ]

        importances = importances.groupby(groups).sum()

    importances = importances.sort_values(ascending=False)

    top_importances = importances.head(top_n).sort_values()

    plt.figure(figsize=(8, max(4, top_n * 0.3)))
    top_importances.plot(kind="barh")
    plt.xlabel(f"Permutation importance ({scoring})")
    plt.ylabel("Residue position" if combine_at_position else "Feature")
    plt.title(f"Top {top_n} permutation importances")
    plt.tight_layout()
    plt.show()

    return importances


def importances_series_to_dict(importances_series, num_features=10):
    output_dict = {}
    c = 0
    for i, j in importances_series.items():
        if c > 15:
            break
        c += 1
        prop, pos = i.split("_")
        if prop not in output_dict:
            output_dict[prop] = []
        output_dict[prop].append(int(pos))

    return output_dict
        
def stdev_accuracy(model, X, y, cv=5, iterations=100):
    """Calculate the standard deviation of accuracy across cross-validation folds."""
    
    accuracies = []
    for i in range(iterations):
        X_eval, y_eval = shuffle(X, y, random_state=i)
        cv = cross_val_score(model, X_eval, y_eval, cv=5)
        accuracies.append(avg(cv))
    return np.mean(accuracies), np.std(accuracies)
