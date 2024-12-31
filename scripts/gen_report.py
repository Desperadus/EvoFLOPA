#!/usr/bin/env python3

import argparse
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Report from Docking Results CSV")
    parser.add_argument(
        "--csv_path", type=str, required=True,
        help="Path to the docking results CSV file."
    )
    parser.add_argument(
        "--top_n", type=int, default=50,
        help="Number of top molecules to include for rmtscoring."
    )

    return parser.parse_args()

def create_plots(df, output_dir):
    sns.set(style="whitegrid")
    
    def compute_metrics(df, column):
        grouped = df.groupby('Iteration')[column]
        metrics_df = pd.DataFrame({
            "Iteration": grouped.mean().index,
            "Top": grouped.max().values,
            "Median": grouped.median().values,
            "Average": grouped.mean().values,
            "Lowest": grouped.min().values
        })
        
        # Apply smoothing using rolling average
        window_size = len(metrics_df) // 10  # Adjust
        if window_size > 0:
            for col in ['Top', 'Median', 'Average', 'Lowest']:
                metrics_df[f'{col}_smooth'] = metrics_df[col].rolling(window=window_size, center=True).mean()
        
        return metrics_df
    
    # Affinity (loss_value) Plot
    loss_value_metrics = compute_metrics(df, 'loss_value')
    plt.figure(figsize=(10, 6))
    # Plot original data with low alpha
    sns.lineplot(data=loss_value_metrics, x='Iteration', y='Top', label='Top', alpha=0.2)
    sns.lineplot(data=loss_value_metrics, x='Iteration', y='Median', label='Median', alpha=0.2)
    sns.lineplot(data=loss_value_metrics, x='Iteration', y='Average', label='Average', alpha=0.2)
    # Plot smoothed data with full opacity
    if 'Top_smooth' in loss_value_metrics.columns:
        plt.plot(loss_value_metrics['Iteration'], loss_value_metrics['Top_smooth'], label='Top (Smoothed)', linewidth=2.5)
        plt.plot(loss_value_metrics['Iteration'], loss_value_metrics['Median_smooth'], label='Median (Smoothed)', linewidth=2.5)
        plt.plot(loss_value_metrics['Iteration'], loss_value_metrics['Average_smooth'], label='Average (Smoothed)', linewidth=2.5)
    plt.title('Loss Progression Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "loss_progression.png")
    plt.close()

    affinity_metrics = compute_metrics(df, 'Docking score')
    plt.figure(figsize=(10, 6))
    # Plot original data with low alpha
    sns.lineplot(data=affinity_metrics, x='Iteration', y='Top', label='Top', alpha=0.2)
    sns.lineplot(data=affinity_metrics, x='Iteration', y='Median', label='Median', alpha=0.2)
    sns.lineplot(data=affinity_metrics, x='Iteration', y='Average', label='Average', alpha=0.2)
    # Plot smoothed data with full opacity
    if 'Top_smooth' in affinity_metrics.columns:
        plt.plot(affinity_metrics['Iteration'], affinity_metrics['Top_smooth'], label='Top (Smoothed)', linewidth=2.5)
        plt.plot(affinity_metrics['Iteration'], affinity_metrics['Median_smooth'], label='Median (Smoothed)', linewidth=2.5)
        plt.plot(affinity_metrics['Iteration'], affinity_metrics['Average_smooth'], label='Average (Smoothed)', linewidth=2.5)
    plt.title('Affinity Progression Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Docking score (Affinity)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "affinity_progression.png")
    plt.close()

    # QED Plot
    qed_metrics = compute_metrics(df, 'QED')
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=qed_metrics, x='Iteration', y='Top', label='Top', alpha=0.2)
    sns.lineplot(data=qed_metrics, x='Iteration', y='Median', label='Median', alpha=0.2)
    sns.lineplot(data=qed_metrics, x='Iteration', y='Average', label='Average', alpha=0.2)
    if 'Top_smooth' in qed_metrics.columns:
        plt.plot(qed_metrics['Iteration'], qed_metrics['Top_smooth'], label='Top (Smoothed)', linewidth=2.5)
        plt.plot(qed_metrics['Iteration'], qed_metrics['Median_smooth'], label='Median (Smoothed)', linewidth=2.5)
        plt.plot(qed_metrics['Iteration'], qed_metrics['Average_smooth'], label='Average (Smoothed)', linewidth=2.5)
    plt.title('QED Progression Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('QED')
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "qed_progression.png")
    plt.close()

    # SA Plot
    sa_metrics = compute_metrics(df, 'SA')
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=sa_metrics, x='Iteration', y='Median', label='Median', alpha=0.2)
    sns.lineplot(data=sa_metrics, x='Iteration', y='Average', label='Average', alpha=0.2)
    sns.lineplot(data=sa_metrics, x='Iteration', y='Lowest', label='Lowest', alpha=0.2)
    if 'Median_smooth' in sa_metrics.columns:
        plt.plot(sa_metrics['Iteration'], sa_metrics['Median_smooth'], label='Median (Smoothed)', linewidth=2.5)
        plt.plot(sa_metrics['Iteration'], sa_metrics['Average_smooth'], label='Average (Smoothed)', linewidth=2.5)
        plt.plot(sa_metrics['Iteration'], sa_metrics['Lowest_smooth'], label='Lowest (Smoothed)', linewidth=2.5)
    plt.title('SA Progression Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('SA')
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "sa_progression.png")
    plt.close()

    print(f"Plots saved in {output_dir}")

def create_network_plot(df, output_dir):
    G = nx.DiGraph()

    seeds = df[['Iteration', 'Seed']].sort_values('Iteration')
    seeds = seeds.dropna().reset_index(drop=True)

    for i in range(1, len(seeds)):
        prev_seed = seeds.loc[i-1, 'Seed']
        current_seed = seeds.loc[i, 'Seed']
        if prev_seed == current_seed:
            continue

        if G.has_edge(prev_seed, current_seed):
            G[prev_seed][current_seed]['weight'] += 1
        else:
            G.add_edge(prev_seed, current_seed, weight=1)

    # Count how often each seed was used
    seed_counts = df['Seed'].value_counts().to_dict()
    for node in G.nodes():
        G.nodes[node]['count'] = seed_counts.get(node, 1)

    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)  # Seed for reproducibility

    node_sizes = [G.nodes[node]['count'] * 2 for node in G.nodes()]
    node_colors = [G.nodes[node]['count'] for node in G.nodes()]

    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]

    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, cmap='viridis', alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, width=weights, alpha=0.5, arrowstyle='->', arrowsize=10, ax=ax)
    # Draw labels
    # nx.draw_networkx_labels(G, pos, font_size=8, font_family="sans-serif", ax=ax)

    sm = plt.cm.ScalarMappable(cmap='viridis', 
                               norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Seed Usage Count')

    plt.title('Seed Molecules Network')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "seed_network_by_usage.png")
    plt.close()

    print(f"Network plot by usage count saved in {output_dir}")

    fig, ax = plt.subplots(figsize=(12, 8))

    affinity_scores = df.groupby('Seed')['loss_value'].mean().to_dict()
    node_colors_affinity = [affinity_scores.get(node, 0) for node in G.nodes()]

    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors_affinity, cmap='coolwarm', alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, width=weights, alpha=0.5, arrowstyle='->', arrowsize=10, ax=ax)

    sm = plt.cm.ScalarMappable(cmap='coolwarm', 
                               norm=plt.Normalize(vmin=min(node_colors_affinity), vmax=max(node_colors_affinity)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Binding Score Affinity')

    plt.title('Seed Molecules Network by Binding Score Affinity')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "seed_network_by_affinity.png")
    plt.close()

    print(f"Network plot by binding score affinity saved in {output_dir}")

def save_top_molecules_to_file(df, file_path):
    with open(file_path, 'w') as file:
        file.write("\nTop 5 Molecules by Loss Value (Affinity):\n")
        top_loss = df.nlargest(5, 'loss_value')[['sdf_path', 'loss_value']]
        file.write(top_loss.to_string(index=False))
        file.write("\n")

        file.write("\nTop 5 Molecules by Docking Score:\n")
        # Assuming lower docking score is better
        top_docking = df.nlargest(5, 'Docking score')[['sdf_path', 'Docking score']]
        file.write(top_docking.to_string(index=False))
        file.write("\n")

        file.write("\nTop 5 Molecules by QED:\n")
        top_qed = df.nlargest(5, 'QED')[['sdf_path', 'QED']]
        file.write(top_qed.to_string(index=False))
        file.write("\n")

        file.write("\nTop 5 Molecules by SA:\n")
        top_sa = df.nsmallest(5, 'SA')[['sdf_path', 'SA']]
        file.write(top_sa.to_string(index=False))
        file.write("\n")


def get_morgan_fingerprint(sdf_path):
    """Generate Morgan fingerprint for a molecule."""
    try:
        mol = Chem.SDMolSupplier(str(sdf_path))[0]
        if mol is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
    except:
        return None

def prepare_best_molecules_for_rmtscoring(df, output_dir, top_n=50):
    os.makedirs(output_dir, exist_ok=True)

    # Sort molecules by loss_value
    sorted_molecules = df.sort_values('loss_value', ascending=False)
    
    # Initialize list to store unique molecules
    unique_molecules = []
    unique_fingerprints = set()
    
    # Iterate through molecules until we find top_n unique ones
    for _, row in sorted_molecules.iterrows():
        if len(unique_molecules) >= top_n:
            break
            
        fp = get_morgan_fingerprint(row['sdf_path'])
        if fp is None:
            continue
            
        fp_bytes = fp.ToBitString()
        if fp_bytes not in unique_fingerprints:
            unique_fingerprints.add(fp_bytes)
            unique_molecules.append(row)

    # Convert to DataFrame
    top_molecules = pd.DataFrame(unique_molecules)[['sdf_path', 'loss_value', 'QED', 'SA', 'Docking score', 'SELFIE']]
    # Add molecule name as best_1, best_2, ...
    top_molecules['molecule_name'] = [f"best_{i}" for i in range(1, len(top_molecules) + 1)]
    top_molecules.to_csv(output_dir / "top_molecules_for_rmtscoring.csv", index=False)

    # Copy unique sdf files to output_dir
    for i, row in enumerate(top_molecules.itertuples(), start=1):
        sdf_path = Path(row.sdf_path)
        new_sdf_path = output_dir / f"best_{i}.sdf"
        os.system(f"cp {sdf_path} {new_sdf_path}")

    print(f"Top {len(top_molecules)} unique molecules by loss value saved in {output_dir}")

    # get list of paths to the new sdf files
    new_sdf_files = [str(output_dir / f"best_{i}.sdf") for i in range(1, len(top_molecules) + 1)]
    return new_sdf_files

def extract_and_combine_best_poses(input_sdf_files, output_sdf_file):
    """
    Extracts the best scoring pose from each input SDF file and writes them to a single output SDF file.

    Parameters:
        input_sdf_files (list of str): List of file paths to input SDF files.
        output_sdf_file (str): Path to the output SDF file.
    """
    def get_best_pose(sdf_file, molecule_name):
        supplier = Chem.SDMolSupplier(sdf_file, removeHs=False)
        best_mol = None
        best_score = float('inf')  # Initialize with a very large number

        for mol in supplier:
            if mol is None:  # Skip invalid molecules
                continue
            score = mol.GetProp("docking_score")  # Extract the docking score
            if float(score) < best_score:
                best_score = float(score)
                best_mol = mol

        if best_mol:
            best_mol.SetProp("_Name", molecule_name)  # Rename the molecule
        return best_mol

    # Create an SDWriter to write the output SDF
    writer = Chem.SDWriter(output_sdf_file)

    for idx, sdf_file in enumerate(input_sdf_files, start=1):
        molecule_name = f"best_{idx}"
        best_pose = get_best_pose(sdf_file, molecule_name)
        if best_pose:
            writer.write(best_pose)  # Write the best scoring pose to the output SDF

    writer.close()
    print(f"Best scoring poses have been concatenated into {output_sdf_file}")

def main():
    args = parse_args()

    csv_path = Path(args.csv_path)
    # Output directory is the same as the CSV file
    output_dir = csv_path.parent / "report"
    os.makedirs(output_dir, exist_ok=True)
    

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    required_columns = {"sdf_path", "loss_value", "QED", "SA", "Docking score", "Iteration", "Seed"}
    if not required_columns.issubset(set(df.columns)):
        print(f"CSV file is missing required columns. Required columns are: {required_columns}")
        return

    create_plots(df, output_dir)
    create_network_plot(df, output_dir)
    save_top_molecules_to_file(df, output_dir / "top_molecules.txt")
    best_mols_paths = prepare_best_molecules_for_rmtscoring(df, output_dir / "top_molecules_for_rmtscoring", top_n=args.top_n)
    extract_and_combine_best_poses(best_mols_paths, output_dir / "top_molecules_for_rmtscoring" / "best_poses.sdf")

    print(f"\nReport generation completed. All plots are saved in '{output_dir}' directory.")

if __name__ == "__main__":
    main()

