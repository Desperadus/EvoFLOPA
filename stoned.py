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

    def generate_molecules(
        self,
        seed_molecule_path: Path,
        output_dir: Path,
        num_molecules: int = 10,
        num_conformers: int = 1
                        ) -> List[Path]:
        """Generates a set of molecules using STONED with conformer optimization and parallel processing."""
        os.makedirs(output_dir, exist_ok=True)

        # Load Initial Molecule
        mols = Chem.SDMolSupplier(str(seed_molecule_path), removeHs=True)
        if not mols or not mols[0]:
            raise Exception(f"Cannot obtain valid molecule using {seed_molecule_path}")
        mol = mols[0]
        

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

                mutated_mol = Chem.MolFromSmiles(mutated_smiles)
                if mutated_mol is None:
                    return None
                
                mutated_mol = Chem.AddHs(mutated_mol)
                mutated_mol.SetProp("SMILES", mutated_smiles)
                mutated_mol.SetProp("SELFIE", mutated_selfie)

                # Add conformers
                for _ in range(num_conformers):
                    embed_status = AllChem.EmbedMolecule(mutated_mol, AllChem.ETKDG())
                    if embed_status != 0:
                        logging.warning(f"Embedding failed for molecule {i+1}")
                        continue
                    AllChem.MMFFOptimizeMolecule(mutated_mol)

                file_name = f"generated_molecule_{i+1}.sdf"
                file_path = output_dir / file_name
                with Chem.SDWriter(str(file_path)) as writer:
                    writer.write(mutated_mol)
                return file_path
            except:
                return None

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_molecule, i) for i in range(num_molecules)]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    generated_molecules.append(result)

        return generated_molecules
