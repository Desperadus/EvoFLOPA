from rdkit import Chem
from rdkit.Chem import Descriptors, QED
from typing import List
from utils import get_docking_score
import logging


def calculate_loss(sdf_file: str, config_data: dict) -> float:
    """Calculates a loss score based on SA, QED, and docking scores."""
    try:
        mol = Chem.SDMolSupplier(sdf_file, removeHs=True)[0]
        if mol is None:
            raise ValueError(f'Cannot get molecule {sdf_file}')
        sa_score = config_data['sa_weight'] * Descriptors.MolLogP(mol)
        qed_score = config_data['qed_weight'] * QED.qed(mol)
        docking_score = config_data["docking_weight"] * \
            get_docking_score(sdf_file)

        loss_value = sa_score + qed_score + docking_score
        return loss_value
    except Exception as e:
        logging.error(f"Error during loss calculation of {sdf_file}: {e}")
        return float('-inf')  # If an error occurs, return the worst loss value
