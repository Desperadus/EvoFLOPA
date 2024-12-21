import random
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from selfies import encoder, decoder, get_semantic_robust_alphabet
from utils import sanitize_smiles
from pathlib import Path
from typing import List
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from random import randrange
import matplotlib.pyplot as plt
import rdkit
from rdkit.Chem import MolFromSmiles as smi2mol
from rdkit.Chem import MolToSmiles as mol2smi
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import Levenshtein
from Levenshtein import distance, editops

from PIL import Image
from rdkit.Chem import RDConfig
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

"""
Parts of code are borrowed from https://github.com/aspuru-guzik-group/stoned-selfies/blob/main/realize_path.py
@author: akshat
"""

class STONED:
    def __init__(self, config_data: dict):
        self.config_data = config_data

    def mutate_selfie(self, selfie: str, max_molecules_len: int) -> tuple[str, str]:
        '''
         Return a mutated selfie string (only one mutation on selfie is performed)

        Rules of mutation: With a 33.3% propbabily, either:
        1. Add a random SELFIE character in the string
        2. Replace a random SELFIE character with another
        3. Delete a random character
        '''
        valid = False
        chars_selfie = self.get_selfie_chars(selfie)

        while not valid:
            # 34 SELFIE characters
            alphabet = list(get_semantic_robust_alphabet())

            choice_ls = [1, 2, 3]  # 1=Insert; 2=Replace; 3=Delete
            random_choice = np.random.choice(choice_ls, 1)[0]

            # Insert a character in a Random Location
            if random_choice == 1:
                random_index = np.random.randint(len(chars_selfie)+1)
                random_character = np.random.choice(alphabet, size=1)[0]

                selfie_mutated_chars = chars_selfie[:random_index] + [
                    random_character] + chars_selfie[random_index:]

            # Replace a random character
            elif random_choice == 2:
                random_index = np.random.randint(len(chars_selfie))
                random_character = np.random.choice(alphabet, size=1)[0]
                if random_index == 0:
                    selfie_mutated_chars = [
                        random_character] + chars_selfie[random_index+1:]
                else:
                    selfie_mutated_chars = chars_selfie[:random_index] + [
                        random_character] + chars_selfie[random_index+1:]

            # Delete a random character
            elif random_choice == 3:
                random_index = np.random.randint(len(chars_selfie))
                if random_index == 0:
                    selfie_mutated_chars = chars_selfie[random_index+1:]
                else:
                    selfie_mutated_chars = chars_selfie[:random_index] + \
                        chars_selfie[random_index+1:]
            else:
                raise Exception('Invalid Operation trying to be performed')

            selfie_mutated = "".join(x for x in selfie_mutated_chars)

            try:
                smiles = decoder(selfie_mutated)
                _, smiles_canon, valid = sanitize_smiles(smiles)
                if len(selfie_mutated_chars) > max_molecules_len or not valid:
                    valid = False
            except:
                valid = False

        return (selfie_mutated, smiles_canon)

    def get_ECFP4(self, mol):
        ''' Return rdkit ECFP4 fingerprint object for mol

        Parameters: 
        mol (rdkit.Chem.rdchem.Mol) : RdKit mol object  

        Returns: 
        rdkit ECFP4 fingerprint object for mol
        '''
        return AllChem.GetMorganFingerprint(mol, 2)

    def sanitize_smiles(self, smi):
        '''Return a canonical smile representation of smi
        
        Parameters:
        smi (string) : smile string to be canonicalized 
        
        Returns:
        mol (rdkit.Chem.rdchem.Mol) : RdKit mol object                          (None if invalid smile string smi)
        smi_canon (string)          : Canonicalized smile representation of smi (None if invalid smile string smi)
        conversion_successful (bool): True/False to indicate if conversion was  successful 
        '''
        try:
            mol = smi2mol(smi, sanitize=True)
            smi_canon = mol2smi(mol, isomericSmiles=False, canonical=True)
            return (mol, smi_canon, True)
        except:
            return (None, None, False)

    def get_fp_scores(self, smiles_back, target_smi): 
        '''Calculate the Tanimoto fingerprint (ECFP4 fingerint) similarity between a list 
        of SMILES and a known target structure (target_smi). 
        
        Parameters:
        smiles_back   (list) : A list of valid SMILES strings 
        target_smi (string)  : A valid SMILES string. Each smile in 'smiles_back' will be compared to this stucture
        
        Returns: 
        smiles_back_scores (list of floats) : List of fingerprint similarities
        '''
        smiles_back_scores = []
        target    = Chem.MolFromSmiles(target_smi)
        fp_target = self.get_ECFP4(target)
        for item in smiles_back: 
            mol    = Chem.MolFromSmiles(item)
            fp_mol = self.get_ECFP4(mol)
            score  = TanimotoSimilarity(fp_mol, fp_target)
            smiles_back_scores.append(score)
        return smiles_back_scores


    def get_selfie_chars(self, selfie: str) -> List[str]:
        '''Obtain a list of all selfie characters in string selfie

        Parameters: 
        selfie (string) : A selfie string - representing a molecule 

        Example: 
        >>> get_selfie_chars('[C][=C][C][=C][C][=C][Ring1][Branch1_1]')
        ['[C]', '[=C]', '[C]', '[=C]', '[C]', '[=C]', '[Ring1]', '[Branch1_1]']

        Returns:
        chars_selfie: list of selfie characters present in molecule selfie
        '''
        chars_selfie = []  # A list of all SELFIE sybols from string selfie
        while selfie != '':
            chars_selfie.append(selfie[selfie.find('['): selfie.find(']')+1])
            selfie = selfie[selfie.find(']')+1:]
        return chars_selfie

    def randomize_smiles(self, mol):
        '''Returns a random (dearomatized) SMILES given an rdkit mol object of a molecule.

        Parameters:
        mol (rdkit.Chem.rdchem.Mol) :  RdKit mol object (None if invalid smile string smi)
        
        Returns:
        mol (rdkit.Chem.rdchem.Mol) : RdKit mol object  (None if invalid smile string smi)
        '''
        if not mol:
            return None

        Chem.Kekulize(mol)
        
        return rdkit.Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=False,  kekuleSmiles=True)


    def get_random_smiles(self, smi, num_random_samples): 
        ''' Obtain 'num_random_samples' non-unique SMILES orderings of smi
        
        Parameters:
        smi (string)            : Input SMILES string (needs to be a valid molecule)
        num_random_samples (int): Number fo unique different SMILES orderings to form 
        
        Returns:
        randomized_smile_orderings (list) : list of SMILES strings
        '''
        mol = Chem.MolFromSmiles(smi)
        if mol == None: 
            raise Exception('Invalid starting structure encountered')
        randomized_smile_orderings  = [self.randomize_smiles(mol) for _ in range(num_random_samples)]
        randomized_smile_orderings  = list(set(randomized_smile_orderings)) # Only consider unique SMILE strings
        return randomized_smile_orderings


    def obtain_path(self, starting_smile, target_smile, filter_path=False): 
        ''' Obtain a path/chemical path from starting_smile to target_smile
        
        Parameters:
        starting_smile (string) : SMILES string (needs to be a valid molecule)
        target_smile (int)      : SMILES string (needs to be a valid molecule)
        filter_path (bool)      : If True, a chemical path is returned, else only a path
        
        Returns:
        path_smiles (list)                  : A list of smiles in path between starting_smile & target_smile
        path_fp_scores (list of floats)     : Fingerprint similarity to 'target_smile' for each smiles in path_smiles
        smiles_path (list)                  : A list of smiles in CHEMICAL path between starting_smile & target_smile (if filter_path==False, then empty)
        filtered_path_score (list of floats): Fingerprint similarity to 'target_smile' for each smiles in smiles_path (if filter_path==False, then empty)
        '''
        starting_selfie = encoder(starting_smile)
        target_selfie = encoder(target_smile)
        
        starting_selfie_chars = self.get_selfie_chars(starting_selfie)
        target_selfie_chars = self.get_selfie_chars(target_selfie)
        
        # Compute edit operations at the token level
        ops = editops(starting_selfie_chars, target_selfie_chars)
        
        path = {}
        path[0] = starting_selfie_chars.copy()
        current_selfie = starting_selfie_chars.copy()
        path_step = 0

        #Debug
        # print(f"Starting SELFIE: {starting_selfie}")
        # print(f"Target SELFIE: {target_selfie}")
        # print(f"Starting SELFIE Chars: {starting_selfie_chars}")
        # print(f"Operations: {ops}")
        
        # Apply edit operations
        offset = 0  # Track index shifts dynamically

        # Apply edit operations
        # shuffle the order of the operations
        for op in ops:
            path_step += 1
            operation, idx_start, idx_target = op

            adjusted_idx_start = idx_start + offset
            
            if operation == 'replace':
                if adjusted_idx_start < len(current_selfie) and idx_target < len(target_selfie_chars):
                    current_selfie[adjusted_idx_start] = target_selfie_chars[idx_target]
            elif operation == 'insert':
                if idx_target < len(target_selfie_chars):
                    current_selfie.insert(adjusted_idx_start, target_selfie_chars[idx_target])
                    offset += 1  # Increment offset due to insertion
            elif operation == 'delete':
                if adjusted_idx_start < len(current_selfie):
                    del current_selfie[adjusted_idx_start]
                    offset -= 1  # Decrement offset due to deletion
            
            # Record the new state
            path[path_step] = current_selfie.copy()
            # print(path_step, current_selfie)
        
        # Collapse path to make them into SELFIE strings
        paths_selfies = []
        for i in range(len(path)):
            selfie_str = ''.join(x for x in path[i] if x != ' ') 
            paths_selfies.append(selfie_str)

        if paths_selfies[-1] != target_selfie: 
            raise Exception("Unable to discover target structure!")
        
        # Obtain similarity scores
        path_smiles = [decoder(x) for x in paths_selfies]
        
        return path_smiles, paths_selfies, None, None


    def get_compr_paths(self, starting_smile, target_smile, num_tries, num_random_samples):
        ''' Obtaining multiple paths/chemical paths from starting_smile to target_smile. 
        
        Parameters:
        starting_smile (string)     : SMILES string (needs to be a valid molecule)
        target_smile (int)          : SMILES string (needs to be a valid molecule)
        num_tries (int)             : Number of path/chemical path attempts between the exact same smiles
        num_random_samples (int)    : Number of different SMILES string orderings to conside for starting_smile & target_smile 
        collect_bidirectional (bool): If true, forms paths from target_smiles-> target_smiles (doubles number of paths)
        
        Returns:
        smiles_paths_dir1 (list): list paths containing smiles in path between starting_smile -> target_smile
        smiles_paths_dir2 (list): list paths containing smiles in path between target_smile -> starting_smile
        '''
        starting_smile_rand_ord = self.get_random_smiles(starting_smile, num_random_samples=num_random_samples)
        target_smile_rand_ord   = self.get_random_smiles(target_smile,   num_random_samples=num_random_samples)
        
        smiles_paths_dir1 = [] # All paths from starting_smile -> target_smile
        selfies_paths_dir1 = [] # All paths from starting_smile -> target_smile
        for smi_start in starting_smile_rand_ord: 
            for smi_target in target_smile_rand_ord: 
                
                if Chem.MolFromSmiles(smi_start) == None or Chem.MolFromSmiles(smi_target) == None: 
                    raise Exception('Invalid structures')
                    
                for _ in range(num_tries): 
                    path_smiles, path_selfies, _, _ = self.obtain_path(smi_start, smi_target, filter_path=False)
                    smiles_paths_dir1.append(path_smiles)
                    selfies_paths_dir1.append(path_selfies)
        
                        
        return smiles_paths_dir1, selfies_paths_dir1

    def get_SA_scores(self, mols): 
        '''Calculate the SA score for a list of molecules
        
        Parameters:
        mols (list) : A list of rdkit mol objects
        
        Returns:
        sa_scores (list of floats) : List of SA scores
        '''
        sa_scores = []
        for mol in mols: 
            sa_scores.append(sascorer.calculateScore(mol))
        return sa_scores
    
    def generate_variations_via_mutations(self, mol, num_conformers, num_molecules, output_dir, all_docked_selfies):
        '''Generate variations of a molecule using mutations
        '''
        if mol.HasProp("SELFIE"):
            starting_selfie = mol.GetProp("SELFIE")
        else:
            # Encode with SELFIE string
            starting_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
            starting_selfie = encoder(starting_smiles)

        # Encode with SELFIE string
        len_random_struct = len(self.get_selfie_chars(starting_selfie))

        generated_molecules = []

        def process_molecule(i):
            try:
                # Mutate the self string
                mutated_selfie, mutated_smiles = self.mutate_selfie(
                    starting_selfie, max_molecules_len=len_random_struct
                )
                if mutated_selfie in all_docked_selfies:
                    # logging.info(f"Skipping molecule with selfie {mutated_selfie} as it has already been docked")
                    return None


                mutated_mol = Chem.MolFromSmiles(mutated_smiles)
                if mutated_mol is None:
                    logging.warning(f"Invalid SMILES for molecule {i+1}: {mutated_smiles}")
                    return None
                
                mutated_mol = Chem.AddHs(mutated_mol)
                mutated_mol.SetProp("SMILES", mutated_smiles)
                mutated_mol.SetProp("SELFIE", mutated_selfie)

                # Check if molecule is valid
                try:
                    Chem.SanitizeMol(mutated_mol)
                except:
                    logging.warning(f"Sanitization failed for molecule {i+1}")
                    return None

                # Add conformers
                conformer_paths = []
                for conf_id in range(num_conformers):
                    embed_status = AllChem.EmbedMolecule(mutated_mol, AllChem.ETKDG())
                    if embed_status != 0:
                        logging.warning(f"Embedding failed for molecule {i+1}, conformer {conf_id+1}")
                        continue
                    AllChem.MMFFOptimizeMolecule(mutated_mol)

                    # Create a copy of the molecule with just this conformer
                    conf_mol = Chem.Mol(mutated_mol)
                    
                    file_name = f"generated_molecule_{i+1}_conf_{conf_id+1}.sdf"
                    file_path = output_dir / file_name
                    with Chem.SDWriter(str(file_path)) as writer:
                        writer.write(conf_mol)
                    conformer_paths.append(file_path)
                
                return conformer_paths if conformer_paths else None
            except:
                logging.warning(f"Failed to generate molecule {i+1}")
                return None

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_molecule, i) for i in range(num_molecules)]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    generated_molecules.extend([path for path in result if path])

        return generated_molecules 


    def generate_children_via_breeding(self, mol1, mol2, num_conformers, num_molecules, output_dir, all_docked_selfies):
        mol1_smiles = Chem.MolToSmiles(mol1, isomericSmiles=True, canonical=True)
        mol2_smiles = Chem.MolToSmiles(mol2, isomericSmiles=True, canonical=True)

        paths_smiles, paths_selfies = self.get_compr_paths(mol1_smiles, mol2_smiles, num_tries=1, num_random_samples=2)
        merged_paths_smiles = [item for sublist in paths_smiles for item in sublist]
        merged_paths_selfies = [item for sublist in paths_selfies for item in sublist]

        mols_from_paths = []    
        for smi, selfie in zip(merged_paths_smiles, merged_paths_selfies):
            mol = Chem.MolFromSmiles(smi)
            try:
                Chem.SanitizeMol(mol)
            except:
                logging.warning(f"Sanitization failed for molecule {i+1}")
                continue
            mol.SetProp("SMILES", smi)
            mol.SetProp("SELFIE", selfie)
            mols_from_paths.append(mol)

        sa_scores = self.get_SA_scores(mols_from_paths)
        # print(sa_scores)
        # Sort by SA score smaller then 4
        sorted_indices = np.argsort(sa_scores)
        mols_from_paths = [mols_from_paths[i] for i in sorted_indices if sa_scores[i] < 4]
        mols_from_paths = mols_from_paths[:min(num_molecules, len(mols_from_paths))]

        def create_conformers(mol, i):
            mol = Chem.AddHs(mol)
            try:
                # Add conformers
                conformer_paths = []
                for conf_id in range(num_conformers):
                    embed_status = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
                    if embed_status != 0:
                        logging.warning(f"Embedding failed for molecule {i+1}, conformer {conf_id+1}")
                        continue
                    AllChem.MMFFOptimizeMolecule(mol)

                    # Create a copy of the molecule with just this conformer
                    conf_mol = Chem.Mol(mol)
                    
                    file_name = f"generated_molecule_{i+1}_conf_{conf_id+1}.sdf"
                    file_path = output_dir / file_name
                    with Chem.SDWriter(str(file_path)) as writer:
                        writer.write(conf_mol)
                    conformer_paths.append(file_path)
                
                return conformer_paths if conformer_paths else None
            except:
                logging.warning(f"Failed to generate molecule {i+1}")
                return None

        generated_molecules = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(create_conformers, mol, i) for i, mol in enumerate(mols_from_paths)]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    generated_molecules.extend([path for path in result if path])

        return generated_molecules

    def generate_molecules(
        self,
        seed_molecule_path: List[Path],
        output_dir: Path,
        num_molecules: int = 32,
        num_conformers: int = 1,
        all_docked_selfies: set[str] = None
                        ) -> List[Path]:
        """Generates a set of molecules using STONED with conformer optimization and parallel processing."""
        os.makedirs(output_dir, exist_ok=True)

        # Load Initial Molecule
        mols = Chem.SDMolSupplier(str(seed_molecule_path[0]), removeHs=True)
        if not mols or not mols[0]:
            raise Exception(f"Cannot obtain valid molecule using {seed_molecule_path}")
        mol1 = mols[0]

        if len(seed_molecule_path) > 1:
            # Load second molecule
            mols = Chem.SDMolSupplier(str(seed_molecule_path[1]), removeHs=True)
            if not mols or not mols[0]:
                raise Exception(f"Cannot obtain valid molecule using {seed_molecule_path}")
            mol2 = mols[0]


        # Generate variations for the first molecule
        if len(seed_molecule_path) == 1:
            generated_molecules = self.generate_variations_via_mutations(
                mol1, num_conformers, num_molecules, output_dir, all_docked_selfies
            )
        else:
            # Generate variations for the second molecule
            generated_molecules = self.generate_children_via_breeding(
                mol1, mol2, num_conformers, num_molecules, output_dir, all_docked_selfies
            )
        return generated_molecules

if __name__ == "__main__":
    # Example usage
    config_data = {
        "sa_weight": 1,
        "qed_weight": 1,
        "docking_weight": 1
    }
    stoned = STONED(config_data)
    seed_molecule_path = [Path("examples/TrmD/Compound23_docked.sdf"), Path("examples/TrmD/Compound37_docked.sdf")]
    output_dir = Path("experiments/generated_molecules")
    generated_molecules = stoned.generate_molecules(seed_molecule_path, output_dir, num_molecules=32, num_conformers=1)
    print(len(generated_molecules))