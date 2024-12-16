import os
import subprocess
from pathlib import Path
from typing import List, Tuple
import json
import logging
from unidock_tools.application.mcdock import MultiConfDock


class Docking:
    def __init__(self, config_data: dict):
        self.config_data = config_data

    def run_unidock_mcdock(
            self,
            receptor_path: Path,
            ligand_paths: List[Path],
            center_coords: tuple,
            box_sizes: tuple,
            output_dir: Path,
            num_modes: int = 1,
            score_only: bool = False
    ) -> List[Path]:

        # Instantiate MultiConfDock
        mcdock = MultiConfDock(
            receptor=receptor_path,
            ligands=ligand_paths,
            center_x=float(center_coords[0]),
            center_y=float(center_coords[1]),
            center_z=float(center_coords[2]),
            size_x=box_sizes[0],
            size_y=box_sizes[1],
            size_z=box_sizes[2],
            workdir=output_dir / "mcdock_tmp_dir"
        )

        mcdock.run_unidock(
            scoring_function="vina",
            num_modes=num_modes,
            score_only=score_only,
        )
        scored_molecules = mcdock.save_results(save_dir=output_dir)
        return scored_molecules
