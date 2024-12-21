import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List
import threading 

from unidock_tools.application.mcdock import MultiConfDock
from unidock_tools.application.unidock_pipeline import UniDock
import config
from loss import calculate_loss
from stoned import STONED

from rdkit import Chem

from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from typing import List, Dict

import csv
import numpy as np


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
    parser.add_argument("-en", "--experiment_name", type=str,
                        default="iterative_docking", help="Working directory.")
    parser.add_argument("-conf", "--config", type=str,
                        default="config.json", help="config_file")
    parser.add_argument("-bs", "--batch_size", type=int, default=512,
                        help="Batch size for docking in unidock")
    parser.add_argument("-nv", "--num_variants", type=int, default=32,
                        help="Number of variants to generate in each iteration.")
    parser.add_argument("-dt", "--docking_threads", type=int, default=4,
                        help="Number of docking tasks to run in parallel.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose mode.")
    parser.add_argument("-tnh", "--top_n_history", type=int, default=25,
                        help="Number of best molecules to consider for selection.")
    parser.add_argument("-temp", "--temperature", type=float, default=0.666,
                        help="Temperature molecule selection")

    return parser


def select_seed_molecule(best_molecules_history, args):
    """
    Select a seed molecule from a list of previously known good molecules.
    The selection probability is weighted by their 'loss_value'.
    A 'temperature' parameter allows tuning the exploration vs exploitation:
    - Lower temperature -> selects top scoring molecules more deterministically.
    - Higher temperature -> more random selection among top performers.
    """

    if not best_molecules_history:
        raise ValueError("No best molecules available for selection.")

    # Ensure loss_value exists for first molecule
    if "loss_value" not in best_molecules_history[0]:
        best_molecules_history[0]["loss_value"] = 0

    # Sort molecules by score (descending)
    sorted_molecules = sorted(best_molecules_history, key=lambda x: x["loss_value"], reverse=True)
    # Consider only the top_k best molecules
    top_mols = sorted_molecules[:args.top_n_history]

    # Extract scores and compute a softmax distribution
    scores = np.array([m["loss_value"] for m in top_mols])
    # Softmax probabilities
    probs = np.exp(scores / args.temperature) / np.sum(np.exp(scores / args.temperature))

    if args.verbose:
        print("Scores:", scores)
        print("Probs:", probs)

    chosen_molecule = np.random.choice(top_mols, p=probs)
    return chosen_molecule


def basic_docking(args, generated_molecules_sdf_list, iteration):
    unidock = UniDock(
        receptor=Path(args.receptor).resolve(),
        ligands=generated_molecules_sdf_list,
        center_x=args.center_x,
        center_y=args.center_y,
        center_z=args.center_z,
        size_x=args.size_x,
        size_y=args.size_y,
        size_z=args.size_z,
        workdir=Path(args.experiment_name).resolve() / "workdir",
    )

    # Perform Docking and Score Evaluation
    unidock.docking(
        scoring_function="vina",
        num_modes=3,
        batch_size=args.batch_size,
        save_dir=Path(args.experiment_name).resolve() / "results" / f"docking_{iteration+1}",
        docking_dir_name=f"docking_{iteration+1}",
    )
    scored_molecules_list = [Path(args.experiment_name).resolve() / "results" / f"docking_{iteration+1}" / f"{Path(mol_path).stem}.sdf" for mol_path in generated_molecules_sdf_list]

    return scored_molecules_list


def score_molecule(scored_mol_path, config_data):
    loss_value, metrics = calculate_loss(sdf_file=scored_mol_path, config_data=config_data)
    return {"sdf_path": scored_mol_path, "loss_value": loss_value, "metrics": metrics}


def score_molecules(scored_molecules_list, config_data):
    molecules_data: List[Dict] = []

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(score_molecule, scored_mol_path, config_data) for scored_mol_path in scored_molecules_list]
        for future in as_completed(futures):
            result = future.result()
            molecules_data.append(result)

    return molecules_data


# Creates a csv for the docking experiment, where each row contains the molecule's name, docking score, QED, SA etc.
def create_docking_csv(csv_path):
    with open(csv_path, mode='w', newline='') as csv_file:
        fieldnames = ["sdf_path", "loss_value", "QED", "SA", "Docking score", "Iteration", "Seed", "SELFIE"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()


def add_docking_csv_results(csv_path, molecules_data, iteration, seed, all_docked_selfies):
    with open(csv_path, mode='a', newline='') as csv_file:
        fieldnames = ["sdf_path", "loss_value", "QED", "SA", "Docking score", "Iteration", "Seed", "SELFIE"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        for molecule_data in molecules_data:
            loss_value = molecule_data["loss_value"]
            QED = molecule_data["metrics"].get("QED", "")
            SA = molecule_data["metrics"].get("SA", "")
            docking_score = molecule_data["metrics"].get("Docking score", "")
            sdf_path = molecule_data["sdf_path"]
            selfie = Chem.SDMolSupplier(sdf_path, removeHs=True)[0].GetProp("SELFIE")
            all_docked_selfies.add(selfie)
            writer.writerow({"sdf_path": sdf_path, "loss_value": loss_value, "QED": QED, "SA": SA, "Docking score": docking_score, "Iteration": iteration, "Seed": seed, "SELFIE": selfie})


def producer_task(stoned, best_molecules_history, generation_output_queue, iteration, all_docked_selfies, args):
    # Get seed molecule before logging to avoid reference error
    seed_mol = select_seed_molecule(best_molecules_history, args)
    logging.info(f"Iteration {iteration}: Generating {args.num_variants} molecules from {seed_mol['sdf_path']}")
    
    output_dir = Path(args.experiment_name).resolve() / "workdir" / f"generation_{iteration+1}"
    
    generated_molecules = stoned.generate_molecules(
        seed_molecule_path=seed_mol["sdf_path"],
        output_dir=output_dir,
        num_molecules=args.num_variants,
        num_conformers=args.num_confs,
        all_docked_selfies=all_docked_selfies
    )
    generation_output_queue.put((iteration, generated_molecules, seed_mol["sdf_path"]))


def consumer_task(generation_output_queue, args, config_data, best_molecules_history, csv_path, iteration, docking_semaphore, all_docked_selfies):
    # Acquire the semaphore before starting docking
    with docking_semaphore:
        # Wait for the next batch of generated molecules
        iteration_generated, generated_molecules_sdf_list, seed_mol = generation_output_queue.get()  # Blocks if empty

        # Perform docking and scoring
        logging.info(f"Iteration {iteration_generated}: Docking {len(generated_molecules_sdf_list)} molecules")
        scored_molecules_list = basic_docking(args, generated_molecules_sdf_list=generated_molecules_sdf_list, iteration=iteration_generated)
        molecules_data = score_molecules(scored_molecules_list, config_data)
        # Note that this function also adds selfie to all_docked_selfies
        add_docking_csv_results(csv_path, molecules_data, iteration, seed_mol, all_docked_selfies)

        # Update best molecule history
        if molecules_data:
            best_current = max(molecules_data, key=lambda m: m["loss_value"])
            if best_current["loss_value"] > min([m["loss_value"] for m in best_molecules_history]):
                best_molecules_history.append(best_current)
            logging.info(f"Iteration {iteration_generated}: Best molecule score {best_current['loss_value']} at {best_current['sdf_path']}, with metrics {best_current['metrics']}")
        else:
            logging.warning(f"Iteration {iteration_generated}: No valid molecules!")


def main():
    parser = get_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,  # Corrected logging level
                        format="%(asctime)s [%(levelname)s] %(message)s")

    # Load configuration file
    config_data = config.load_config(args.config)

    # Initialize parameters
    workdir = Path(args.experiment_name).resolve() / "workdir"
    savedir = Path(args.experiment_name).resolve() / "results"
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(savedir, exist_ok=True)
    create_docking_csv(Path(savedir / "docking_results.csv"))

    stoned = STONED(config_data=config_data)
    all_docked_selfies = set() # Set of docked molecules to avoid duplicates

    # Initialize bounded queue
    generation_output_queue = Queue(maxsize=8)

    best_molecules_history = [{
        "sdf_path": args.ligand,
        "loss_value": 0,
        "metrics": {}
    }]

    # Initialize semaphore to limit docking tasks to
    docking_semaphore = threading.Semaphore(args.docking_threads)

    with ThreadPoolExecutor() as executor:
        # Launch initial producer task for iteration 0
        executor.submit(producer_task, stoned, best_molecules_history, generation_output_queue, 0, all_docked_selfies, args)

        for i in range(args.num_iterations):
            # Launch consumer task for docking
            executor.submit(
                consumer_task,
                generation_output_queue,
                args,
                config_data,
                best_molecules_history,
                Path(savedir / "docking_results.csv"),
                i,
                docking_semaphore,  # Pass the semaphore to consumer_task
                all_docked_selfies
            )

            # Launch next producer task for iteration i+1
            if i < args.num_iterations - 1:
                executor.submit(producer_task, stoned, best_molecules_history, generation_output_queue, i+1, all_docked_selfies, args)

    logging.info("Optimization finished!")


if __name__ == "__main__":
    main()
