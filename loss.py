from rdkit import Chem
from rdkit.Chem import QED
from typing import List
from utils import get_docking_score
import logging

from rdkit.Chem import RDConfig
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

#suppress rdkit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def calculate_loss(sdf_file: str, config_data: dict) -> float:
    """Calculates a loss score based on SA, QED, and docking scores."""
    try:
        mol = Chem.SDMolSupplier(sdf_file, removeHs=True)[0]
        if mol is None:
            raise ValueError(f'Cannot get molecule {sdf_file}')

        real_sa_score = sascorer.calculateScore(mol)
        real_qed_score = QED.qed(mol)
        selfie = mol.GetProp('SELFIE') 

        sa_score = config_data['sa_weight'] * (1 - real_sa_score/10)
        qed_score = config_data['qed_weight'] * real_qed_score
        docking_score = config_data["docking_weight"] * \
            get_docking_score(sdf_file)*-1

        loss_value = sa_score + qed_score + docking_score
        return loss_value, {"SA": real_sa_score, "QED": real_qed_score, "Docking score": docking_score, "Adjusted SA": sa_score, "Adjusted QED": qed_score, "SELFIE": selfie}
    except Exception as e:
        logging.error(f"Error during loss calculation of {sdf_file}: {e}")
        return float('-inf'), {"SA": float('-inf'), "QED": float('-inf'), "Docking score": float('-inf'), "Adjusted SA": float('-inf'), "Adjusted QED": float('-inf'), "SELFIE": ""}
