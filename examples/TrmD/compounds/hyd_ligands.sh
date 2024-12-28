#!/bin/bash

# List of input ligand files
ligands=(
  "Compound21_docked.sdf"
  "Compound23_docked.sdf"
  "Compound35_docked.sdf"
  "Compound36_docked.sdf"
  "Compound37_docked.sdf"
  "CompoundX_docked.sdf"
  "CompoundY_docked.sdf"
)

# Loop through each ligand and hydrogenate
for ligand in "${ligands[@]}"; do
  # Extract the base name without the extension
  base_name="${ligand%.sdf}"

  # Add hydrogen at pH 7.4 and save with new name
  obabel "$ligand" -O "${base_name}_hydrogenated.sdf" --ph 7.4 -h
done

echo "Hydrogenation completed for all ligands."
