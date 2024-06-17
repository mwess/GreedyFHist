"""
Contains io helper functions
"""
import json
import os
from os.path import exists
from pathlib import Path
import shutil
from typing import Dict, Optional

import numpy
import numpy as np
import SimpleITK
import SimpleITK as sitk
import yaml


def create_if_not_exists(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def clean_if_exists(path):
    if os.path.exists(path):
        shutil.rmtree(path)

def write_coordinates_as_vtk(coordinates: numpy.array, output_path: str) -> None:
    """
    Writes 2d coordinates in vtk format.
    :param coordinates:
    :param output_path:
    :return:
    """
    # Deal with inf values.
    content = ['# vtk DataFile Version 2.0', 'Landmark test', 'ASCII', 'DATASET POLYDATA',
               f'POINTS {coordinates.shape[0]} float']
    for i in range(coordinates.shape[0]):
        x = "{:.6f}".format(coordinates[i, 0])
        y = "{:.6f}".format(coordinates[i, 1])
        if x == 'inf':
            x = '0'
        if y == 'inf':
            y = '0'
        content.append(f'{x}\t{y} 0')
    content = '\n'.join(content)
    with open(output_path, 'w') as f:
        f.write(content)


def read_simple_vtk(path: str) -> numpy.array:
    """
    Reads vtk from path. No suitable for general vtk reading.
    :param path:
    :return:
    """
    with open(path) as f:
        content = f.read().split('\n')
        # Header is the first 5 lines. Skip those
        content = ' '.join(content[5:]).split()
        point_list = []
        for i in range(0, len(content), 3):
            x = float(content[i])
            y = float(content[i + 1])
            # Since we are only interested in x and y we skip the z coordinate.
            # z = float(content[i+2])
            point_list.append((x, y))
        return np.array(point_list)

def write_mat_to_file(mat: numpy.array, fname: str) -> None:
    out_str = f'{mat[0,0]} {mat[0,1]} {mat[0,2]}\n{mat[1,0]} {mat[1,1]} {mat[1,2]}\n{mat[2,0]} {mat[2,1]} {mat[2,2]}'
    with open(fname, 'w') as f:
        f.write(out_str)

def read_sitk_if_not_none(path: str) -> Optional[SimpleITK.SimpleITK.Image]:
    if exists(path):
        return sitk.GetArrayFromImage(sitk.ReadImage(path))
    return None


# TODO: Remove?!
def read_config(path: str) -> Dict:
    with open(path) as f:
        if path.endswith('.json'):
            return json.load(f)
        elif path.endswith('.yaml'):
            return yaml.safe_load(f)