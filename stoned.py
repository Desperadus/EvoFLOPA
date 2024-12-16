import random
import numpy as np
from rdkit import Chem
from selfies import encoder, decoder, get_semantic_robust_alphabet
from utils import sanitize_smiles
from pathlib import Path
from typing import List


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

    def generate_molecules(self, seed_molecule_path: Path, output_dir: Path, num_molecules: int = 10) -> List[Path]:
        """Generates a set of molecules using STONED."""
        os.makedirs(output_dir, exist_ok=True)

        # Load Initial Molecule
        mols = Chem.SDMolSupplier(str(seed_molecule_path), removeHs=True)
        if not mols or not mols[0]:
            raise Exception(
                f"Cannot obtain valid molecule using {seed_molecule_path}")
        mol = mols[0]
        starting_smiles = Chem.MolToSmiles(
            mol, isomericSmiles=True, canonical=True)

        # Encode with SELFIE string
        starting_selfie = encoder(starting_smiles)
        len_random_struct = len(self.get_selfie_chars(starting_selfie))

        generated_molecules = []
        for i in range(num_molecules):
            # Mutate the self string and save as .sdf file
            mutated_selfie, mutated_smiles = self.mutate_selfie(
                starting_selfie, max_molecules_len=len_random_struct)

            try:
                mutated_mol = Chem.MolFromSmiles(mutated_smiles)
            except:
                raise ValueError("Invalid SMILES encountered after mutation")

            if mutated_mol == None:
                raise ValueError(
                    "Invalid rdkit mol object created after SELFIE mutation")

            file_name = f"generated_molecule_{i+1}.sdf"
            file_path = Path(output_dir / file_name)
            with Chem.SDWriter(str(file_path)) as writer:
                writer.write(mutated_mol)
            generated_molecules.append(file_path)

        return generated_molecules
