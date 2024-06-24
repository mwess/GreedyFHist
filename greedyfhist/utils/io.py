"""
Contains io helper functions
"""
import os
from os.path import join
from pathlib import Path
import shutil

import numpy


def create_if_not_exists(path: str) -> None:
    """Creates directory, if it does not exist. Clashes with existing directories result in nothing getting created.

    Args:
        path (str):
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def clean_if_exists(path: str):
    """Remove directory, if it exists.

    Args:
        path (str): 
    """
    if os.path.exists(path):
        shutil.rmtree(path)


def write_mat_to_file(mat: numpy.array, fname: str) -> None:
    """Writes 2d affine matrix to file. File format is human readable.

    Args:
        mat (numpy.array): 
        fname (str):
    """
    out_str = f'{mat[0,0]} {mat[0,1]} {mat[0,2]}\n{mat[1,0]} {mat[1,1]} {mat[1,2]}\n{mat[2,0]} {mat[2,1]} {mat[2,2]}'
    with open(fname, 'w') as f:
        f.write(out_str)


def derive_output_path(directory: str, fname: str, limit: int = 1000) -> str:
    """Generates a unique output path. If path is already existing,
    adds a counter value until a unique path is found.

    Args:
        directory (str): target directory
        fname (str): target filename
        limit (int, optional): Limit number to prevent endless loops. Defaults to 1000.

    Returns:
        str: Target path
    """
    target_path = join(directory, fname)
    if not os.path.exists(target_path):
        return target_path
    for suffix in range(limit):
        new_target_path = f'{target_path}_{suffix}'
        if not os.path.exists(new_target_path):
            return new_target_path
    return target_path