import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List

from unidock_tools.application.mcdock import MultiConfDock
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
    parser.add_argument("--num_iterations", type=int, default=5,
                        help="Number of iterations for the iterative docking.")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Number of molecules to generate in each iteration.")
    parser.add_argument("-wd", "--workdir", type=str,
                        default="iterative_docking_workdir", help="Working directory.")
    parser.add_argument("-sd", "--savedir", type=str,
                        default="iterative_docking_results", help="Save directory.")
    parser.add_argument("-conf", "--config", type=str,
                        default="config.json", help="config_file")

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
    batch_size = args.batch_size
    workdir = Path(args.workdir).resolve()
    savedir = Path(args.savedir).resolve()

    # Instantiate STONED
    stoned = STONED(config_data=config_data)

    # Initialize Uni-Dock
    mcdock = MultiConfDock(
        receptor=receptor_path,
        ligands=[ligand_path],
        center_x=center_coords[0],
        center_y=center_coords[1],
        center_z=center_coords[2],
        size_x=box_sizes[0],
        size_y=box_sizes[1],
        size_z=box_sizes[2],
        workdir=workdir,
    )

    best_molecule_path = ligand_path
    # Initialize with negative infinity (assuming we want to maximize the score)
    best_score = float('-inf')

    for i in range(num_iterations):
        logging.info(f"Starting iteration {i+1}/{num_iterations}")

        # 1. Generate molecules with STONED
        generated_molecules_sdf_list = stoned.generate_molecules(
            seed_molecule_path=best_molecule_path,
            output_dir=Path(workdir / f"generation_{i+1}").resolve(),
            num_molecules=batch_size
        )
        # 2. Perform Docking and Score Evaluation
        mcdock.run_unidock(
            scoring_function="vina",
            num_modes=1,
            score_only=True,
            batch_size=1,
            docking_dir_name=f"docking_{i+1}",
        )
        scored_molecules = mcdock.save_results(
            save_dir=Path(savedir / f"docking_{i+1}").resolve())

        # Each entry contains docking, QED, SA etc. scores.
        molecules_data: List[Dict] = []
        for scored_mol_path in scored_molecules:
            loss_value = calculate_loss(
                sdf_file=scored_mol_path, config_data=config_data)

            molecules_data.append(
                {"sdf_path": scored_mol_path,
                 "loss_value": loss_value}
            )

        # 3. Find best performing molecule from the current batch
        if molecules_data:
            best_current_molecule = max(
                molecules_data, key=lambda mol: mol["loss_value"])
            if best_current_molecule["loss_value"] > best_score:
                best_score = best_current_molecule["loss_value"]
                best_molecule_path = best_current_molecule["sdf_path"]
            logging.info(
                f" Best molecule obtained with score {best_score} is in {best_molecule_path}")

        else:
            logging.warning(
                f"No molecules were generated that satisfy validity criteria or have valid docking scores!")

    logging.info("Optimization finished!")


if __name__ == "__main__":
    main()
