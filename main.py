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
    parser.add_argument("-nv", "--num_variants", type=int, default=10,
                        help="Number of variants to generate in each iteration.")
    parser.add_argument("-dp", "--docking_paralellism", type=int, default=4,
                        help="Number of docking tasks to run in parallel.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose mode.")
    parser.add_argument("-tnh", "--top_n_history", type=int, default=100,
                        help="Number of best molecules to consider for selection.")

    return parser


def select_seed_molecule(best_molecules_history, top_k=100, temperature=1.0, verbose=False):
    """
    Select a seed molecule from a list of previously known good molecules.
    The selection probability is weighted by their 'loss_value'.
    A 'temperature' parameter allows tuning the exploration vs exploitation:
    - Lower temperature -> selects top scoring molecules more deterministically.
    - Higher temperature -> more random selection among top performers.
    """

    if not best_molecules_history:
        raise ValueError("No best molecules available for selection.")

    # Sort molecules by score (descending)
    sorted_molecules = sorted(best_molecules_history, key=lambda x: x["loss_value"], reverse=True)
    # Consider only the top_k best molecules
    top_mols = sorted_molecules[:top_k]

    # Extract scores and compute a softmax distribution
    scores = np.array([m["loss_value"] for m in top_mols])
    # Softmax probabilities
    probs = np.exp(scores / temperature) / np.sum(np.exp(scores / temperature))

    if verbose:
        print("Scores:", scores)
        print("Probs:", probs)

    chosen_molecule = np.random.choice(top_mols, p=probs)
    return chosen_molecule


def basic_docking(receptor_path, ligand_path, center_coords, box_sizes, workdir, savedir, num_confs, batch_size, generated_molecules_sdf_list, iteration):
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

    # Perform Docking and Score Evaluation
    unidock.docking(
        scoring_function="vina",
        num_modes=3,
        batch_size=batch_size,
        save_dir=Path(savedir / f"docking_{iteration+1}").resolve(),
        docking_dir_name=f"docking_{iteration+1}",
    )
    scored_molecules_list = [Path(savedir / f"docking_{iteration+1}" / f"{Path(mol_path).stem}.sdf") for mol_path in generated_molecules_sdf_list]

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
        fieldnames = ["sdf_path", "loss_value", "QED", "SA", "Docking score"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()


def add_docking_csv_results(csv_path, molecules_data):
    with open(csv_path, mode='a', newline='') as csv_file:
        fieldnames = ["sdf_path", "loss_value", "QED", "SA", "Docking score"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        for molecule_data in molecules_data:
            loss_value = molecule_data["loss_value"]
            QED = molecule_data["metrics"].get("QED", "")
            SA = molecule_data["metrics"].get("SA", "")
            docking_score = molecule_data["metrics"].get("Docking score", "")
            sdf_path = molecule_data["sdf_path"]
            writer.writerow({"sdf_path": sdf_path, "loss_value": loss_value, "QED": QED, "SA": SA, "Docking score": docking_score})


def producer_task(stoned, best_molecules_history, generation_output_queue, num_variants, num_confs, iteration, workdir):
    try:
        # Select a seed molecule from history
        seed_mol = select_seed_molecule(best_molecules_history)
    except ValueError as e:
        logging.error(f"Producer Task Error: {e}")
        return

    output_dir = workdir / f"generation_{iteration+1}"
    generated_molecules = stoned.generate_molecules(
        seed_molecule_path=seed_mol["sdf_path"],
        output_dir=output_dir,
        num_molecules=num_variants,
        num_conformers=num_confs
    )
    # Put generated molecules into a queue for docking
    generation_output_queue.put((iteration, generated_molecules))


def docking_task(receptor_path, ligand_path, center_coords, box_sizes, workdir, savedir, num_confs, batch_size, molecule_list, iteration):
    scored_molecules_list = basic_docking(
        receptor_path=receptor_path,
        ligand_path=ligand_path,
        center_coords=center_coords,
        box_sizes=box_sizes,
        workdir=workdir,
        savedir=savedir,
        num_confs=num_confs,
        batch_size=batch_size,
        generated_molecules_sdf_list=molecule_list,
        iteration=iteration
    )
    return scored_molecules_list


def consumer_task(generation_output_queue, docking_params, config_data, best_molecules_history, csv_path, iteration, docking_semaphore):
    # Acquire the semaphore before starting docking
    with docking_semaphore:
        # Wait for the next batch of generated molecules
        iteration_generated, generated_molecules_sdf_list = generation_output_queue.get()  # Blocks if empty
        logging.info(f"Consumer: Docking molecules for iteration {iteration_generated}, queue size: {generation_output_queue.qsize()}")

        # Perform docking and scoring
        logging.info(f"Iteration {iteration_generated}: Docking {len(generated_molecules_sdf_list)} molecules")
        scored_molecules_list = basic_docking(**docking_params, generated_molecules_sdf_list=generated_molecules_sdf_list, iteration=iteration_generated)
        molecules_data = score_molecules(scored_molecules_list, config_data)
        add_docking_csv_results(csv_path, molecules_data)

        # Update best molecule history
        
        if molecules_data:
            best_current = max(molecules_data, key=lambda m: m["loss_value"])
            if best_current["loss_value"] > best_molecules_history[-1]["loss_value"]:
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

    # Initialize bounded queue
    generation_output_queue = Queue(maxsize=8)

    docking_params = {
        "receptor_path": Path(args.receptor).resolve(),
        "ligand_path": Path(args.ligand).resolve(),
        "center_coords": (args.center_x, args.center_y, args.center_z),
        "box_sizes": (args.size_x, args.size_y, args.size_z),
        "workdir": workdir,
        "savedir": savedir,
        "num_confs": args.num_confs,
        "batch_size": args.batch_size
    }

    best_molecules_history = [{
        "sdf_path": args.ligand,
        "loss_value": 0,
        "metrics": {}
    }]

    # Initialize semaphore to limit docking tasks to
    docking_semaphore = threading.Semaphore(args.docking_paralellism)

    with ThreadPoolExecutor() as executor:
        # Launch initial producer task for iteration 0
        executor.submit(producer_task, stoned, best_molecules_history, generation_output_queue,
                        args.num_variants, args.num_confs, 0, workdir)

        for i in range(args.num_iterations):
            # Launch consumer task for docking
            executor.submit(
                consumer_task,
                generation_output_queue,
                docking_params,
                config_data,
                best_molecules_history,
                Path(savedir / "docking_results.csv"),
                i,
                docking_semaphore  # Pass the semaphore to consumer_task
            )

            # Launch next producer task for iteration i+1
            if i < args.num_iterations - 1:
                executor.submit(producer_task, stoned, best_molecules_history, generation_output_queue,
                                args.num_variants, args.num_confs, i+1, workdir)

    logging.info("Optimization finished!")


if __name__ == "__main__":
    main()
