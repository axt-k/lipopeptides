"""Cheminformatics analysis of acyl starter structures."""

import argparse 
import logging
import os
import typing as ty
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.decomposition import PCA

# Use Arial font for plots
plt.rcParams["font.family"] = "Arial"


# Setup logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


# Define constants
ALPHA_GREEK = "\u03B1"
PALETTE = [
    "#e69f00",
    "#56b4e9",
    "#039e73",
    "#f0e442",
    "#0072b2",
    "#d55f00",
    "#cc79a7",
    "#000000",
    "#808285",
]
SANCTIONED_LABELS = [
    "AR",  # Aromatic
    # "FA",  # Fatty acid; commented out as there are none in the dataset
    "LCFA",  # Long-chain fatty acid
    "MCFA",  # Medium-chain fatty acid
    "SCFA",  # Short-chain fatty acid
]
PATTERN_COOH = Chem.MolFromSmarts("C(=O)[OH]")  # COOH group pattern


def category_to_color(category: str) -> str:
    """Map a category to a color.

    :param category: Category name.
    :return: Hex color code for the category.
    """
    category_to_color = {
        "AR": PALETTE[0],
        "FA": PALETTE[1],
        "LCFA": PALETTE[2],
        "MCFA": PALETTE[3],
        "SCFA": PALETTE[4],
    }

    return category_to_color.get(category, "#000000")  # Default to black if category not found


def cli() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    :return: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument("--data", "-d", type=Path, required=True, help="path to the input data csv file")
    parser.add_argument("--out-dir", "-o", type=Path, required=True, help="output directory to save the results")

    # Optional arguments
    parser.add_argument("--cname-starter-smiles", "-s", type=str, default="starter_canonical_smiles", help="column name for starter SMILES in the input data (default: 'starter_canonical_smiles')")
    parser.add_argument("--cname-starter-category", "-c", type=str, default="starter_category", help="column name for starter category in the input data (default: 'starter_category')")
    parser.add_argument("--cname-genus", "-g", type=str, default="ncbi_organism_genus", help="column name for genus in the input data (default: 'ncbi_organism_genus')")
    parser.add_argument("--cname-species", "-p", type=str, default="ncbi_organism_species", help="column name for species in the input data (default: 'ncbi_organism_species')")

    parser.add_argument("--log-level", "-l", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="set the logging level (default: 'INFO')")

    return parser.parse_args()


def mol_has_cooh_group(mol: Chem.Mol) -> bool:
    """
    Check if a molecule contains a COOH group.

    :param mol: RDKit molecule object.
    :return: True if the molecule contains a COOH group, False otherwise.
    """
    return mol.HasSubstructMatch(PATTERN_COOH)


class SMILESParseError(Exception):
    """Custom exception for SMILES parsing errors."""
    def __init__(self, message: str) -> None:
        super().__init__(message)


class COOHGroupNotFoundError(Exception):
    """Custom exception for when a COOH group is not found in a molecule."""
    def __init__(self, message: str) -> None:
        super().__init__(message)


@dataclass
class DataRecord:
    """Data record for a single starter structure."""
    starter_smiles: str
    category_starter: str
    genus_producing_organism: str
    species_producing_organism: str
    starter_mol: Chem.Mol = field(init=False)

    def __post_init__(self):
        self.starter_mol = Chem.MolFromSmiles(self.starter_smiles)
        if self.starter_mol is None:
            raise SMILESParseError(f"Failed to parse SMILES: {self.starter_smiles}")
        
        # Ensure molecule contains COOH group
        if not mol_has_cooh_group(self.starter_mol):
            raise COOHGroupNotFoundError(f"COOH group not found in category {self.category_starter} starter with SMILES: {self.starter_smiles}")


def parse_records_from_csv(
    path: str,
    cname_starter_smiles: str,
    cname_starter_category: str,
    cname_genus: str,
    cname_species: str,
) -> ty.List[DataRecord]:
    """
    Parse records from a CSV file.
    
    :param path: Path to the CSV file.
    :param cname_starter_smiles: Column name for starter SMILES.
    :param cname_starter_category: Column name for starter category.
    :param cname_genus: Column name for genus of producing organism.
    :param cname_species: Column name for species of producing organism.
    :return: List of DataRecord objects.
    """
    LOGGER.info(f"Parsing records from {path}")

    # Read the CSV file into a DataFrame
    df = pd.read_csv(path, dtype=str)

    # Only keep where 'ncbi_organism_superkingdom' is 'Bacteria'
    df = df[df["ncbi_organism_superkingdom"] == "Bacteria"]

    # Check if required columns are present
    required_columns = [cname_starter_smiles, cname_starter_category, cname_genus, cname_species]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the input data file.")
        
    # Create DataRecord objects from the DataFrame
    records = []
    for _, row in df.iterrows():
        try:
            LOGGER.debug(f"Record:\n\tstarter_smiles={row[cname_starter_smiles]}\n\tcategory_starter={row[cname_starter_category]}\n\tgenus_producing_organism={row[cname_genus]}\n\tspecies_producing_organism={row[cname_species]}\n")
            record = DataRecord(
                starter_smiles=row[cname_starter_smiles],
                category_starter=row[cname_starter_category],
                genus_producing_organism=row[cname_genus],
                species_producing_organism=row[cname_species]
            )
        except (SMILESParseError, TypeError) as e:
            LOGGER.warning(f"Skipping row due to SMILES parsing error: {e}")
            continue
        except COOHGroupNotFoundError as e:
            LOGGER.warning(f"Skipping row due to missing COOH group: {e}")
            continue
        records.append(record)

    LOGGER.info(f"Parsed {len(records)} records from {path}")
    return records


def plot_starter_pca(out_dir: str, ax: plt.Axes, records: list[DataRecord]) -> None:
    """
    Function to generate the PCA plot for starter structures on a specific axis.
    
    :param out_dir: Directory to save the PCA plot.
    :param ax: Matplotlib Axes object to plot on.
    :param records: List of DataRecord objects containing starter structures.
    """
    fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = [fingerprint_generator.GetFingerprint(record.starter_mol) for record in records]
    labels_category = np.array([record.category_starter for record in records])

    # Convert fingerprints to numpy array
    X = np.zeros((len(fingerprints), 2048))
    for i, fp in enumerate(fingerprints):
        DataStructs.ConvertToNumpyArray(fp, X[i])
    LOGGER.info(f"fingerprints shape: {X.shape}")

    # Perform PCA on the fingerprints
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X)
    LOGGER.info(f"principal components shape: {pcs.shape}")
    explained_variance = pca.explained_variance_ratio_
    LOGGER.info(f"explained variance: {explained_variance}")

    # Plot the individual points
    for label in SANCTIONED_LABELS:
        if label not in SANCTIONED_LABELS:
            continue

        jitter = 0.025
        x = pcs[labels_category == label, 0]
        y = pcs[labels_category == label, 1]
        color = category_to_color(label)

        # Add a bit of jitter to x and y values so that they are not overlapping.
        x += np.random.normal(0, jitter, x.shape)
        y += np.random.normal(0, jitter, y.shape)

        ax.scatter(
            x, y, 
            c=color, 
            s=120,
            label=f"{label} ({len(x)})",
            edgecolor="black", 
            linewidth=0.5,
            zorder=2
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(f"PC 1 ({explained_variance[0] * 100:.2f}%; 2.5% jitter added for scatter plot)", fontsize=16)
    ax.set_ylabel(f"PC 2 ({explained_variance[1] * 100:.2f}%; 2.5% jitter added for scatter plot)", fontsize=16)
    ax.legend(title="Category", title_fontsize="16", fontsize="14")

    # Create separate figure for the PCA plot with numbered axes, save to separate file
    fig_pca = plt.figure(figsize=(8, 6))
    ax_pca = fig_pca.add_subplot(111)
    ax_pca.scatter(pcs[:, 0], pcs[:, 1], c="black", s=10, alpha=0.5)
    ax_pca.set_xlabel(f"PC 1 ({explained_variance[0] * 100:.2f}%)", fontsize=16)
    ax_pca.set_ylabel(f"PC 2 ({explained_variance[1] * 100:.2f}%)", fontsize=16)
    plt.tight_layout()
    path = os.path.join(out_dir, "starter_pca_plot.png")
    plt.savefig(path, dpi=300)
    plt.close(fig_pca)

    return pcs


def bfs_carbon_paths(graph: dict[int, list[int]], graph_identity: dict[int, str], start_node: int) -> tuple[dict[int, float], dict[int, list[int]]]:
    """
    Perform a breadth-first search (BFS) to find the shortest paths from the start node
    to all other carbon atoms in the graph.

    :param graph: A dictionary mapping each node to its list of neighboring node indices.
    :param graph_identity: A dictionary mapping each node index to its atomic symbol.
    :param start_node: The starting node index for the BFS, must correspond to a carbon atom ('C').
    :return: A tuple containing:
             - carbon_distances: A dictionary mapping carbon node indices to their BFS distance from start_node.
             - carbon_paths: A dictionary mapping carbon node indices to the shortest path (list of node indices) from start_node.
    """
    # Check if the start node is carbon
    if graph_identity[start_node] != 'C':
        return {}, {}  # No paths if the start node is not carbon

    queue = deque([start_node])
    distances = {node: float('inf') for node in graph}
    distances[start_node] = 0
    shortest_paths = {node: [] for node in graph}
    shortest_paths[start_node] = [start_node]
    
    while queue:
        current_node = queue.popleft()
        
        for neighbor in graph[current_node]:
            # Only consider carbon atoms for the path
            if graph_identity[neighbor] == 'C' and distances[neighbor] == float('inf'):
                distances[neighbor] = distances[current_node] + 1
                shortest_paths[neighbor] = shortest_paths[current_node] + [neighbor]
                queue.append(neighbor)
    
    # Filter out non-carbon atoms from the results
    carbon_distances = {node: dist for node, dist in distances.items() if graph_identity[node] == 'C'}
    carbon_paths = {node: path for node, path in shortest_paths.items() if graph_identity[node] == 'C'}
    
    return carbon_distances, carbon_paths


def find_longest_shortest_path_unweighted(
    graph: dict[int, list[int]], 
    graph_identity: dict[int, str], 
    start_node: int
) -> list[int]:
    """
    Find the longest shortest path in an unweighted graph starting from a given node.

    This function performs a breadth-first search (BFS) to determine the shortest paths
    from the start_node to all other carbon atoms and then returns the longest of these paths.

    :param graph: A dictionary mapping each node to a list of neighbor node indices.
    :param graph_identity: A dictionary mapping each node index to its atomic symbol.
    :param start_node: The starting node index for the BFS; must correspond to a carbon atom ('C').
    :return: The longest shortest path as a list of node indices.
    """
    _, shortest_paths = bfs_carbon_paths(graph, graph_identity, start_node)
    
    # Find the longest shortest path by comparing path lengths.
    longest_shortest_path = max(shortest_paths.values(), key=len)
    return longest_shortest_path


def retrieve_backbone(mol: Chem.Mol) -> list[int]:
    """
    Retrieve the backbone of a molecule, starting from the alpha carbon (COOH group).
    
    :param mol: RDKit molecule object.
    :return: List of atom indices representing the backbone starting from the alpha carbon.
    :raises COOHGroupNotFoundError: If the molecule does not contain a COOH group.
    """
    graph_identity = {}  # atom idx -> atom symbol
    graph = {}
    for atom in mol.GetAtoms():
        atom_id = atom.GetIdx()
        atom_symbol = atom.GetSymbol()
        graph_identity[atom_id] = atom_symbol
        atom_neighbors = [neighbor.GetIdx() for neighbor in atom.GetNeighbors()]
        graph[atom_id] = atom_neighbors

    # Find atom indices of COOH group.
    pattern = Chem.MolFromSmarts("C(=O)[OH]")
    matches = mol.GetSubstructMatches(pattern)
    if len(matches) == 0:
        raise COOHGroupNotFoundError("COOH group not found in the molecule.")
    alpha_carbon_index = matches[0][0]
    
    # Find the longest shortest path from the alpha carbon.
    longest_shortest_path = find_longest_shortest_path_unweighted(graph, graph_identity, alpha_carbon_index)
    return longest_shortest_path


def index_cstarter_functional_groups(
    ax: plt.Axes,
    mols: list[Chem.Mol],
    backbones: list[list[int]],
    category: str,
    max_height: int,
    max_backbone_len: int,
    set_xaxis_label: bool = False
) -> None:
    """
    Index functional groups along the molecule backbones and plot the results as a stacked bar chart.

    :param ax: Matplotlib Axes on which to plot.
    :param mols: List of RDKit Mol objects.
    :param backbones: List of backbones, where each backbone is a list of atom indices.
    :param category: Category name used for labeling the legend.
    :param max_height: Maximum height for the y-axis scale (determined externally).
    :param max_backbone_len: Maximum length of the backbone across all molecules.
    :param set_xaxis_label: Whether to set the x-axis label explicitly. Defaults to False.
    :return: None.
    """
    label_colors: dict[str, str] = {
        "Carboxylic acid": "#e69f00",
        "Hydroxyl": "#56b4e9",
        "Methyl": "#0072b2",
        "Epoxide": "#d55f00",
        "Ketone": "#039e73",
        "Amine": "#cc79a7",
        "Imine": "#f0e442",
    }
    counts: list[list[int]] = [[0 for _ in range(len(label_colors) + 1)] for _ in range(max_backbone_len)]
    
    for mol, backbone in zip(mols, backbones):
        for i, atom in enumerate(backbone):
            pattern = "C(=O)[OH]"
            matches = mol.GetSubstructMatches(Chem.MolFromSmarts(pattern))
            if matches and atom == matches[0][0]:
                counts[i][0] += 1

            pattern = "[C][CH1]([OH])[C]"
            matches = mol.GetSubstructMatches(Chem.MolFromSmarts(pattern))
            for match in matches:
                if atom == match[1]:
                    counts[i][1] += 1

            pattern = "[C][CH1]([CH3])[C]"
            matches = mol.GetSubstructMatches(Chem.MolFromSmarts(pattern))
            for match in matches:
                if atom == match[1]:
                    counts[i][2] += 1

            pattern = "C1CO1"
            matches = mol.GetSubstructMatches(Chem.MolFromSmarts(pattern))
            for match in matches:
                if atom == match[0]:
                    counts[i][3] += 1

            pattern = "[C][C](=O)[C]"
            matches = mol.GetSubstructMatches(Chem.MolFromSmarts(pattern))
            for match in matches:
                if atom == match[1]:
                    counts[i][4] += 1

            pattern = "[C][NH2]"
            matches = mol.GetSubstructMatches(Chem.MolFromSmarts(pattern))
            for match in matches:
                if atom == match[0]:
                    counts[i][5] += 1
                    # Print SMILES of molecules with amine at position 1 (index 0)
                    if i == 0:
                        smiles = Chem.MolToSmiles(mol)
                        LOGGER.info(f"Molecule with amine at position 1: {smiles}")

            pattern = "[C][NH][C]=[N]"
            matches = mol.GetSubstructMatches(Chem.MolFromSmarts(pattern))
            for match in matches:
                if atom == match[0]:
                    counts[i][6] += 1

            counts[i][7] += 1

    # Plot line plot for backbone lengths
    ax.plot(
        range(1, max_backbone_len + 1),
        [x[7] for x in counts],
        color="black",
        marker="o",
        label="Length",
        markersize=3
    )
    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
    ax.grid(axis="x", linestyle="--", alpha=0.3, zorder=0)

    if set_xaxis_label:
        ax.set_xticks(
            [i for i in range(1, max_backbone_len + 1)],
            ["1"] + [str(i) if i % 5 == 0 else "" for i in range(1, max_backbone_len)],
            fontsize=11
        )
    else:
        ax.set_xticks([i for i in range(1, max_backbone_len + 1)])
        ax.set_xticklabels(
            ["" for _ in range(1, max_backbone_len + 1)],
            fontsize=11
        )
    
    y_tick_distance: int = 1
    if 10 < max_height <= 100:
        y_tick_distance = 10
    elif 100 < max_height <= 1000:
        y_tick_distance = 20
    ax.set_yticks(
        [i for i in range(0, int(1.2 * max_height), y_tick_distance)],
        [str(i) for i in range(0, int(1.2 * max_height), y_tick_distance)],
        fontsize=11
    )
    # Plot stacked bar chart for functional group counts
    bottom: list[int] = [0 for _ in range(max_backbone_len)]
    for i, label in enumerate(label_colors.keys()):
        if any(x[i] > 0 for x in counts):
            total: int = sum(x[i] for x in counts)
            ax.bar(
                range(1, max_backbone_len + 1),
                [x[i] for x in counts],
                label=f"{label} ({total})",
                bottom=bottom,
                color=label_colors[label],
                edgecolor="black",
                zorder=100
            )
            bottom = [sum(x) for x in zip(bottom, [x[i] for x in counts])]

    ax.set_ylim(0, 1.1 * max_height)
    if set_xaxis_label:
        ax.set_xlabel(f"Backbone position (starting at carbonyl carbon)", fontsize=16)

    ax.legend(
        markerfirst=False,
        title_fontsize="16",
        fontsize="14",
        title=f"{category} properties"
    )


def main() -> None:
    """Entry point script."""

    # Parse command line arguments
    args = cli()

    out_dir = args.out_dir
    out_dir.mkdir(exist_ok=True, parents=True)

    # Update logger level
    LOGGER.setLevel(getattr(logging, args.log_level.upper()))

    # Set up logging to file if specified and remove old log file if it exists
    log_file = out_dir / "log.txt"
    if os.path.exists(log_file):
        os.remove(log_file)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, args.log_level.upper()))
    LOGGER.addHandler(file_handler)

    LOGGER.info(f"Command line arguments: {args}")

    # Parse data file
    records = parse_records_from_csv(
        path=args.data,
        cname_starter_smiles=args.cname_starter_smiles,
        cname_starter_category=args.cname_starter_category,
        cname_genus=args.cname_genus,
        cname_species=args.cname_species
    )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Retrieve backbones from starter structures, including their lengths
    # and indices (used for assigning location to functionl groups)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Calculate max backbone lengths
    starter_mols = [mol.starter_mol for mol in records]
    labels_category = np.array([mol.category_starter for mol in records])

    backbones, backbone_mols, backbone_carbons = [], [], []
    for mol, label in zip(starter_mols, labels_category):
        if label in SANCTIONED_LABELS:
            backbone_inds = retrieve_backbone(mol)
            backbones.append(backbone_inds)
            if label != "AR":
                backbone_carbon_count = str(Chem.MolToSmiles(mol)).upper().count("C")
                backbone_carbons.append(backbone_carbon_count)

            backbone_mols.append(mol)
        else:
            backbones.append([])
    # backbone_lens = [len(backbone) for backbone in backbones]
    backbone_lens = [backbone for backbone in backbone_carbons]

    max_backbone_len = max(backbone_lens)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Setup figure grid
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(4, 3)
    # Make space between row 0 and row 1 smaller, only for columns 0 and 1
    sub_gs1 = gs[:, 0:2].subgridspec(4, 2, hspace=0.0)
    # Make row space column 2 smaller
    sub_gs2 = gs[:, 2].subgridspec(4, 1, hspace=0.0)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Column 0, row 0: Bond order plot of backbone
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Get bond order plot of backbone.
    counted = 0
    bond_orders = [[] for _ in range(max_backbone_len)]
    for mol in backbone_mols:
        counted += 1
        backbone = retrieve_backbone(mol)
        for i, atom in enumerate(backbone[:-1]):
            next_atom = backbone[i+1]
            bond = mol.GetBondBetweenAtoms(atom, next_atom)
            bond_order = bond.GetBondTypeAsDouble()
            bond_orders[i].append(bond_order)

    bond_orders_1_heights = []
    bond_orders_2_heights = []
    bond_orders_3_heights = []
    for bin in bond_orders:
        count_1 = bin.count(1.0)
        count_2 = bin.count(2.0)
        count_3 = bin.count(3.0)
        bond_orders_1_heights.append(count_1)
        bond_orders_2_heights.append(count_2)
        bond_orders_3_heights.append(count_3)

    ax8 = fig.add_subplot(gs[0, 0])
    ax8.bar(range(1, max_backbone_len + 1), bond_orders_2_heights, color="#23aae1", edgecolor="black", label=f"Double bonds ({sum(bond_orders_2_heights)})", bottom=[0 for _ in range(max_backbone_len)], facecolor=PALETTE[-2])
    ax8.bar(range(1, max_backbone_len + 1), bond_orders_1_heights, color="#f9e11b", edgecolor="black", label=f"Single bonds ({sum(bond_orders_1_heights)})", bottom=[sum(x) for x in zip(bond_orders_2_heights, bond_orders_3_heights)], facecolor="#ceccca")
    ax8.set_ylim(0, 1.1 * max([sum(x) for x in zip(bond_orders_1_heights, bond_orders_2_heights, bond_orders_3_heights)]))
    ax8.set_xticks(
        [0.5] + [i + 0.5 for i in range(1, max_backbone_len+1)], 
        ["1"] + [str(i + 1) if (i + 1) % 5 == 0 else "" for i in range(0, max_backbone_len)],
        fontsize=11
    )
    ax8.set_yticks(
        [i for i in range(0, int(1.1 * max([sum(x) for x in zip(bond_orders_1_heights, bond_orders_2_heights, bond_orders_3_heights)])), 50)],
        [str(i) for i in range(0, int(1.1 * max([sum(x) for x in zip(bond_orders_1_heights, bond_orders_2_heights, bond_orders_3_heights)])), 50)],
        fontsize=11
    )

    # Change order of labels in legend: single > double
    handles, labels = ax8.get_legend_handles_labels()
    order = [1, 0]
    ax8.legend([handles[idx] for idx in order], [labels[idx] for idx in order], fontsize=14, title_fontsize=16, title="Saturation")

    ax8.grid(axis="y", linestyle="--", alpha=0.5)
    
    ax8.set_xlabel(f"Backbone position (starting at {ALPHA_GREEK}-carbon)", fontsize=16)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Column 1, row 0: Backbone length distribution
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Make bar plot of backbone lengths.
    ax7 = fig.add_subplot(gs[0, 1])
    ax7.hist(backbone_lens, bins=range(1, max_backbone_len+2), edgecolor="black", facecolor="#ceccca", zorder=100)
    max_bin_size = max([backbone_lens.count(i) for i in range(1, max_backbone_len+1)])
    ax7.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax7.set_xticks(
        [i + 0.5 for i in range(1, max_backbone_len+1)], 
        ["1"] + [str(i) if i % 5 == 0 else "" for i in range(2, max_backbone_len+1)],
        fontsize=11
    )
    ax7.set_yticks(
        [i for i in range(0, int(1.1 * max_bin_size), 20)],
        [str(i) for i in range(0, int(1.1 * max_bin_size), 20)],
        fontsize=11
    )
    ax7.set_xlabel(f"Backbone size (number of carbons)", fontsize=16)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Column 0-1, row 1-2: PCA plot of all starter structures
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Plot PCA for all starter structures
    ax1 = fig.add_subplot(sub_gs1[1:4, 0:2])
    pcs = plot_starter_pca(out_dir, ax1, records)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Column 2, row 1-3: Indexing of functional groups in starter structures
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Index LCFA starters.
    ax3 = fig.add_subplot(sub_gs2[3, 0])
    index_cstarter_functional_groups(
        ax3,
        [m for m, l in zip(starter_mols, labels_category) if l == "LCFA"],
        [b for b, l in zip(backbones, labels_category) if l == "LCFA"],
        "LCFA",
        len([c for c in labels_category if c == "LCFA"]),
        max_backbone_len,
        set_xaxis_label=True
    )

    # Index MCFA starters.
    ax4 = fig.add_subplot(sub_gs2[2, 0])
    index_cstarter_functional_groups(
        ax4,
        [m for m, l in zip(starter_mols, labels_category) if l == "MCFA"],
        [b for b, l in zip(backbones, labels_category) if l == "MCFA"],
        "MCFA",
        len([c for c in labels_category if c == "MCFA"]),
        max_backbone_len,
        set_xaxis_label=False
    )

    # Index SCFA starters.
    ax5 = fig.add_subplot(sub_gs2[1, 0])
    index_cstarter_functional_groups(
        ax5,
        [m for m, l in zip(starter_mols, labels_category) if l == "SCFA"],
        [b for b, l in zip(backbones, labels_category) if l == "SCFA"],
        "SCFA",
        len([c for c in labels_category if c == "SCFA"]),
        max_backbone_len,
        set_xaxis_label=False
    )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Report on AR structures
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Get all structures for AR, calculate InChIKey and determine prevalence
    ar_mols = [m for m, l in zip(starter_mols, labels_category) if l == "AR"]
    # Retrieve the indices of AR molecules in the overall list (to map to PCA coordinates)
    ar_indices = [i for i, l in enumerate(labels_category) if l == "AR"]
    # Get PC coordinates for AR molecules from the PCA results
    ar_pc_coords = [pcs[i] for i in ar_indices]

    ar_inchikeys = [Chem.MolToInchiKey(m).split("-")[0] for m in ar_mols]
    ar_inchikey_counts = {k: ar_inchikeys.count(k) for k in set(ar_inchikeys)}
    ar_inchikey_counts = dict(sorted(ar_inchikey_counts.items(), key=lambda x: x[1], reverse=True))

    # Get top 5 InChIKeys and their counts, and one associated molecule (turned into SMILES)
    top_5_inchikeys = list(ar_inchikey_counts.keys())[:5]
    top_5_counts = [ar_inchikey_counts[k] for k in top_5_inchikeys]
    top_5_mols = [ar_mols[ar_inchikeys.index(k)] for k in top_5_inchikeys]
    top_5_smiles = [Chem.MolToSmiles(m) for m in top_5_mols]
    
    # For each top InChIKey, average the PC coordinates of all matching molecules
    top_5_pc_coords = []
    for k in top_5_inchikeys:
        matching_coords = [ar_pc_coords[i] for i, inchikey in enumerate(ar_inchikeys) if inchikey == k]
        matching_coords = np.array(matching_coords)
        avg_coord = matching_coords.mean(axis=0)
        top_5_pc_coords.append(avg_coord)
    
    for k, c, s, pc in zip(top_5_inchikeys, top_5_counts, top_5_smiles, top_5_pc_coords):
        LOGGER.info(f"AR InChIKey: {k}, count: {c}, SMILES: {s}, PC1: {pc[0]:.2f}, PC2: {pc[1]:.2f}")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Output PCA coordinates with SMILES to CSV
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Create a DataFrame with all PCA points mapped to SMILES and categories
    pca_data = []
    for i, (mol, category) in enumerate(zip(starter_mols, labels_category)):
        smiles = Chem.MolToSmiles(mol)
        pc1, pc2 = pcs[i]
        pca_data.append({
            'SMILES': smiles,
            'Category': category,
            'PC1': pc1,
            'PC2': pc2
        })
    
    pca_df = pd.DataFrame(pca_data)
    pca_csv_path = out_dir / "pca_coordinates_with_smiles.csv"
    pca_df.to_csv(pca_csv_path, index=False)
    LOGGER.info(f"PCA coordinates with SMILES saved to {pca_csv_path}")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save plot
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Save the entire figure grid to a file
    plt.subplots_adjust(hspace=0.8, wspace=0.15)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    path_out_png = out_dir / "cstarter_cheminformatics_analysis.png"
    path_out_svg = out_dir / "cstarter_cheminformatics_analysis.svg"
    plt.savefig(path_out_png, dpi=600, transparent=False)
    plt.savefig(path_out_svg, transparent=False)
    plt.close(fig)
    LOGGER.info("Done")
    

if __name__ == "__main__":
    main()
