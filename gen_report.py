#!/usr/bin/env python3

import argparse
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Report from Docking Results CSV")
    parser.add_argument(
        "--csv_path", type=str, required=True,
        help="Path to the docking results CSV file."
    )

    return parser.parse_args()

def create_plots(df, output_dir):
    sns.set(style="whitegrid")
    
    # Function to compute metrics per iteration
    def compute_metrics(df, column):
        grouped = df.groupby('Iteration')[column]
        metrics_df = pd.DataFrame({
            "Iteration": grouped.mean().index,
            "Top": grouped.max().values,
            "Median": grouped.median().values,
            "Average": grouped.mean().values
        })
        return metrics_df
    
    # Affinity (loss_value) Plot
    affinity_metrics = compute_metrics(df, 'loss_value')
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=affinity_metrics, x='Iteration', y='Top', label='Top')
    sns.lineplot(data=affinity_metrics, x='Iteration', y='Median', label='Median')
    sns.lineplot(data=affinity_metrics, x='Iteration', y='Average', label='Average')
    plt.title('Affinity Progression Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Loss Value (Affinity)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "affinity_progression.png")
    plt.close()

    # QED Plot
    qed_metrics = compute_metrics(df, 'QED')
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=qed_metrics, x='Iteration', y='Top', label='Top')
    sns.lineplot(data=qed_metrics, x='Iteration', y='Median', label='Median')
    sns.lineplot(data=qed_metrics, x='Iteration', y='Average', label='Average')
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
    sns.lineplot(data=sa_metrics, x='Iteration', y='Top', label='Top')
    sns.lineplot(data=sa_metrics, x='Iteration', y='Median', label='Median')
    sns.lineplot(data=sa_metrics, x='Iteration', y='Average', label='Average')
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

    # Extract seed relationships
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

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)  # Seed for reproducibility

    # Node sizes based on seed usage count
    node_sizes = [G.nodes[node]['count'] * 2 for node in G.nodes()]
    # Node colors based on seed usage count
    node_colors = [G.nodes[node]['count'] for node in G.nodes()]

    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]

    # Draw nodes
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, cmap='viridis', alpha=0.8, ax=ax)
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=weights, alpha=0.5, arrowstyle='->', arrowsize=10, ax=ax)
    # Draw labels
    # nx.draw_networkx_labels(G, pos, font_size=8, font_family="sans-serif", ax=ax)

    # Create a ScalarMappable for the colorbar
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

    # Create figure and axes for binding score affinity
    fig, ax = plt.subplots(figsize=(12, 8))

    # Node colors based on binding score affinity
    affinity_scores = df.groupby('Seed')['loss_value'].mean().to_dict()
    node_colors_affinity = [affinity_scores.get(node, 0) for node in G.nodes()]

    # Draw nodes
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors_affinity, cmap='coolwarm', alpha=0.8, ax=ax)
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=weights, alpha=0.5, arrowstyle='->', arrowsize=10, ax=ax)

    # Create a ScalarMappable for the colorbar
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

def main():
    args = parse_args()

    csv_path = Path(args.csv_path)
    # Output directory is the same as the CSV file
    output_dir = csv_path.parent / "report"
    os.makedirs(output_dir, exist_ok=True)
    

    # Read CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Validate required columns
    required_columns = {"sdf_path", "loss_value", "QED", "SA", "Docking score", "Iteration", "Seed"}
    if not required_columns.issubset(set(df.columns)):
        print(f"CSV file is missing required columns. Required columns are: {required_columns}")
        return

    # Create plots
    create_plots(df, output_dir)

    # Create network plot
    create_network_plot(df, output_dir)

    # Print top molecules
    save_top_molecules_to_file(df, output_dir / "top_molecules.txt")

    print(f"\nReport generation completed. All plots are saved in '{output_dir}' directory.")

if __name__ == "__main__":
    main()

