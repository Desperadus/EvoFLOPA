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

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def get_parser():
    parser = argparse.ArgumentParser(
        description="Iterative Docking with STONED")

    parser.add_argument("-r", "--receptor", type=str,
                        required=True, help="Path to the receptor PDB file.")
    parser.add_argument("-l", "--ligand", type=str,
                        help="Path to the starting ligand SDF file.")
    # Path to multiple ligands
    parser.add_argument("-ls", "--ligands", type=str, nargs='+',
                        help="Paths to the starting ligands SDF files.")
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
    parser.add_argument("--num_modes", type=int, default=3,
                        help="Number of modes for docking.")
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
    parser.add_argument("-tnh", "--top_n_history", type=int, default=125,
                        help="Number of best molecules to consider for selection.")
    parser.add_argument("-temp", "--temperature", type=float, default=0.666,
                        help="Temperature molecule selection")
    parser.add_argument("--breed", action="store_true",
                        help="Whether to breed molecules or not.")
    parser.add_argument("--breeding_prob", type=float, default=0.5,
                        help="Probability of breeding molecules.")

    return parser


def select_seed_molecule(best_molecules_history, num_of_molecules, args):
    """
    Select a seed molecule from a list of previously known good molecules.
    The selection probability is weighted by their 'loss_value'.
    A 'temperature' parameter allows tuning the exploration vs exploitation:
    - Lower temperature -> selects top scoring molecules more deterministically.
    - Higher temperature -> more random selection among top performers.
    """

    if not best_molecules_history:
        raise ValueError("No best molecules available for selection.")

    # Ensure all molecules have a 'loss_value'; default to 0 if missing
    for molecule in best_molecules_history:
        if "loss_value" not in molecule:
            molecule["loss_value"] = 0

    # Sort molecules by 'loss_value' in descending order
    sorted_molecules = sorted(best_molecules_history, key=lambda x: x["loss_value"], reverse=True)

    # Consider only the top_n_history best molecules
    top_mols = sorted_molecules[:args.top_n_history]

    if num_of_molecules > len(top_mols):
        num_of_molecules = len(top_mols)

    # Extract 'loss_value' scores
    scores = np.array([m["loss_value"] for m in top_mols])

    # Apply temperature scaling and compute softmax probabilities
    scaled_scores = scores / args.temperature
    # To prevent overflow, subtract the max scaled score from all scaled scores
    scaled_scores -= np.max(scaled_scores)
    exp_scores = np.exp(scaled_scores)
    probs = exp_scores / np.sum(exp_scores)

    if num_of_molecules == 1:
        # Select a single molecule based on the computed probabilities
        chosen_molecule = np.random.choice(top_mols, p=probs)
        return [chosen_molecule]
    else:
        # Select multiple unique molecules without replacement
        chosen_molecules = np.random.choice(top_mols, size=num_of_molecules, replace=False, p=probs)
        return list(chosen_molecules)


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
        num_modes=args.num_modes,
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
            selfie = molecule_data["metrics"].get("SELFIE", "")
            all_docked_selfies.add(selfie)
            writer.writerow({"sdf_path": sdf_path, "loss_value": loss_value, "QED": QED, "SA": SA, "Docking score": docking_score, "Iteration": iteration, "Seed": seed, "SELFIE": selfie})


def generate_new_molecules(stoned, best_molecules_history, iteration, all_docked_selfies, args):
    # Get seed molecule before logging to avoid reference error
    # Decide if we should mutate from seed or breed two molecules

    if args.breed:
        if np.random.rand() < args.breeding_prob:
            seed_mols = select_seed_molecule(best_molecules_history, 2, args)
        else:
            seed_mols = select_seed_molecule(best_molecules_history, 1, args)
    else:
        seed_mols = select_seed_molecule(best_molecules_history, 1, args)

    output_dir = Path(args.experiment_name).resolve() / "workdir" / f"generation_{iteration+1}"
    output_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

    generated_molecules = stoned.generate_molecules(
        seed_molecule_path=[molecule["sdf_path"] for molecule in seed_mols],
        output_dir=output_dir,
        num_molecules=args.num_variants,
        num_conformers=args.num_confs,
        all_docked_selfies=all_docked_selfies
    )
    return generated_molecules, seed_mols[0]


def producer_task(stoned, best_molecules_history, generation_output_queue, iteration, all_docked_selfies, args):
    try:
        logging.info(f"Iteration {iteration}: Generating {args.num_variants} molecules, queue size {generation_output_queue.qsize()}")
        generated_molecules, seed_mol = generate_new_molecules(stoned, best_molecules_history, iteration, all_docked_selfies, args)
        generation_output_queue.put((iteration, generated_molecules, seed_mol["sdf_path"]))
    except Exception as e:
        logging.error(f"Producer task failed at iteration {iteration}: {e}")


def consumer_task(generation_output_queue, args, config_data, best_molecules_history, csv_path, docking_semaphore, all_docked_selfies, producer_consumer_semaphore):
    try:
        with docking_semaphore:
            iteration_generated, generated_molecules_sdf_list, seed_mol = generation_output_queue.get()  # Blocks if empty

            logging.info(f"Iteration {iteration_generated}: Docking {len(generated_molecules_sdf_list)} molecules")
            scored_molecules_list = basic_docking(args, generated_molecules_sdf_list=generated_molecules_sdf_list, iteration=iteration_generated)
            molecules_data = score_molecules(scored_molecules_list, config_data)
            # Note that this function also adds selfie to all_docked_selfies
            add_docking_csv_results(csv_path, molecules_data, iteration_generated, seed_mol, all_docked_selfies)

            # Update best molecule history
            if molecules_data:
                best_current = max(molecules_data, key=lambda m: m["loss_value"])
                if best_current["loss_value"] > min([m["loss_value"] for m in best_molecules_history]):
                    best_molecules_history.append(best_current)
                logging.info(f"Iteration {iteration_generated}: Best molecule score {best_current['loss_value']} at {best_current['sdf_path']}, with metrics {best_current['metrics']}")
            else:
                logging.warning(f"Iteration {iteration_generated}: No valid molecules!")
    except Exception as e:
        logging.error(f"Consumer task failed: {e}")
    finally:
        # Release the semaphore to allow a new producer-consumer pair
        producer_consumer_semaphore.release()


def main():
    parser = get_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,  # Corrected logging level
                        format="%(asctime)s [%(levelname)s] %(message)s")
    if args.ligand is None and args.ligands is None:
        raise ValueError("Provide either a single ligand or multiple ligands for the experiment.")

    # Load configuration file
    config_data = config.load_config(args.config)

    # Initialize parameters
    workdir = Path(args.experiment_name).resolve() / "workdir"
    savedir = Path(args.experiment_name).resolve() / "results"
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(savedir, exist_ok=True)
    create_docking_csv(Path(savedir / "docking_results.csv"))

    stoned = STONED(config_data=config_data)
    all_docked_selfies = set()  # Set of docked molecules to avoid duplicates

    # Initialize bounded queue
    generation_output_queue = Queue(maxsize=args.docking_threads*2)

    if args.ligand:
        best_molecules_history = [{
            "sdf_path": args.ligand,
            "loss_value": 10,
            "metrics": {}
        }]
    else:
        best_molecules_history = [{
            "sdf_path": ligand,
            "loss_value": 10,
            "metrics": {}
        } for ligand in args.ligands]

    # Initialize semaphore to limit docking tasks to
    docking_semaphore = threading.Semaphore(args.docking_threads)

    # Initialize semaphore to limit producer-consumer pairs to args.docking_threads*2
    producer_consumer_semaphore = threading.Semaphore(args.docking_threads*2)

    with ThreadPoolExecutor() as executor:
        for i in range(args.num_iterations):
            # Acquire semaphore before submitting a new producer-consumer pair
            producer_consumer_semaphore.acquire()

            # Submit producer task
            executor.submit(
                producer_task,
                stoned,
                best_molecules_history,
                generation_output_queue,
                i,
                all_docked_selfies,
                args
            )

            # Submit consumer task, passing the semaphore to release it after processing
            executor.submit(
                consumer_task,
                generation_output_queue,
                args,
                config_data,
                best_molecules_history,
                Path(savedir / "docking_results.csv"),
                docking_semaphore,
                all_docked_selfies,
                producer_consumer_semaphore
            )

    logging.info("Optimization finished!")


if __name__ == "__main__":
    main()
