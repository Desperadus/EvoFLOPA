import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List

from unidock_tools.application.mcdock import MultiConfDock
from unidock_tools.application.unidock_pipeline import UniDock
import config
from loss import calculate_loss
from stoned import STONED


def get_parser():
    parser = argparse.ArgumentParser(
        description="Iterative Docking with STONED")

    parser.add_argument("-r", "--receptor", type=str,
                        required=True, help="Path to the receptor PDB file.")
    parser.add_argument("-l", "--ligand", type=str, required=True,
                        help="Path to the starting ligand SDF file.")
    parser.add_argument("-cx", "--center_x", type=float, required=True,
                        help="Center X coordinate for the docking box.")
    parser.add_argument("-cy", "--center_y", type=float, required=True,
                        help="Center Y coordinate for the docking box.")
    parser.add_argument("-cz", "--center_z", type=float, required=True,
                        help="Center Z coordinate for the docking box.")
    parser.add_argument("-sx", "--size_x", type=float, default=22.5,
                        help="Size X for docking box (default 22.5)")
    parser.add_argument("-sy", "--size_y", type=float, default=22.5,
                        help="Size Y for docking box (default 22.5)")
    parser.add_argument("-sz", "--size_z", type=float, default=22.5,
                        help="Size Z for docking box (default 22.5)")
    parser.add_argument("--num_iterations", type=int, default=10,
                        help="Number of iterations for the iterative docking.")
    parser.add_argument("--num_confs", type=int, default=1,
                        help="Number of molecules to generate in each iteration.")
    parser.add_argument("-wd", "--workdir", type=str,
                        default="iterative_docking_workdir", help="Working directory.")
    parser.add_argument("-sd", "--savedir", type=str,
                        default="iterative_docking_results", help="Save directory.")
    parser.add_argument("-conf", "--config", type=str,
                        default="config.json", help="config_file")
    parser.add_argument("-bs", "--batch_size", type=int, default=512,
                        help="Batch size for docking in unidock")
    parser.add_argument("-nv", "--num_variants", type=int, default=10,
                        help="Number of variants to generate in each iteration.")

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    # Load configuration file
    config_data = config.load_config(args.config)

    # Initialize parameters
    receptor_path = Path(args.receptor).resolve()
    ligand_path = Path(args.ligand).resolve()
    center_coords = (args.center_x, args.center_y, args.center_z)
    box_sizes = (args.size_x, args.size_y, args.size_z)
    num_iterations = args.num_iterations
    num_variants = args.num_variants
    batch_size = args.batch_size
    num_confs = args.num_confs
    workdir = Path(args.workdir).resolve()
    savedir = Path(args.savedir).resolve()

    # Instantiate STONED
    stoned = STONED(config_data=config_data)

    # Initialize Uni-Dock
    best_molecule = {"sdf_path": ligand_path,
                 "loss_value": float('-inf'),
                    "metrics": {}}

    for i in range(num_iterations):
        logging.info(f"Starting iteration {i+1}/{num_iterations}")

        # 1. Generate molecules with STONED
        generated_molecules_sdf_list = stoned.generate_molecules(
            seed_molecule_path=best_molecule["sdf_path"],
            output_dir=Path(workdir / f"generation_{i+1}").resolve(),
            num_molecules=num_variants,
            num_conformers=num_confs,
        )

        unidock = UniDock(
            receptor=receptor_path,
            ligands=generated_molecules_sdf_list,
            center_x=center_coords[0],
            center_y=center_coords[1],
            center_z=center_coords[2],
            size_x=box_sizes[0],
            size_y=box_sizes[1],
            size_z=box_sizes[2],
            workdir=workdir,
        )

        # 2. Perform Docking and Score Evaluation
        unidock.docking(
            scoring_function="vina",
            num_modes=3,
            batch_size=batch_size,
            save_dir = Path(savedir / f"docking_{i+1}").resolve(),
            docking_dir_name=f"docking_{i+1}",
        )
        scored_molecules_list  =  [Path(savedir / f"docking_{i+1}" / f"{Path(mol_path).stem}.sdf") for mol_path in generated_molecules_sdf_list]

        # Each entry contains docking, QED, SA etc. scores.
        molecules_data: List[Dict] = []
        for scored_mol_path in scored_molecules_list:
            loss_value, metrics = calculate_loss(
                sdf_file=scored_mol_path, config_data=config_data)

            molecules_data.append(
                {"sdf_path": scored_mol_path,
                 "loss_value": loss_value,
                    "metrics": metrics}
            )

        # 3. Find best performing molecule from the current batch
        if molecules_data:
            best_current_molecule = max(
                molecules_data, key=lambda mol: mol["loss_value"])
            if best_current_molecule["loss_value"] > best_molecule["loss_value"]:
                best_molecule = best_current_molecule
            logging.info(
                f" Best molecule obtained with score {best_molecule["loss_value"]} is in {best_molecule["sdf_path"]}, with metrics: {best_molecule['metrics']}")

        else:
            logging.warning(
                f"No molecules were generated that satisfy validity criteria or have valid docking scores!")

    logging.info("Optimization finished!")


if __name__ == "__main__":
    main()
