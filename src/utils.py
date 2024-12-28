import logging
from rdkit import Chem


def sanitize_smiles(smi):
    """Return a canonical SMILES, and if it is not valid return None"""
    try:
        mol = Chem.MolFromSmiles(smi, sanitize=True)
        smi_canon = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
        return (mol, smi_canon, True)
    except:
        return (None, None, False)


def get_docking_score(sdf_file: str) -> float:
    """Extracts docking score from the SD file."""
    try:
        mol = Chem.SDMolSupplier(sdf_file, removeHs=True)[0]
        if not mol:
            raise ValueError(f"Cannot obtain molecules from {sdf_file}")
        docking_score = float(mol.GetDoubleProp('docking_score'))
        return docking_score
    except Exception as e:
        logging.error(f"Cannot get docking score of {sdf_file}, error {e}")
        return 0.0
