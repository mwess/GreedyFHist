"""
Stepwise multisection integration. Uses precomputed transformations.
Registration works by "tarzaning" along each section.
"""
import sys
sys.path.append('../../spami')
sys.path.append('../')
from itertools import permutations
import json
import os
from os.path import join, exists
import shutil
import time

import numpy as np
import pandas as pd
import SimpleITK as sitk

from greedyfhist.registration import greedy_f_hist

from miit.reg_graph import RegGraph
from miit.utils.utils import create_if_not_exists
from greedyfhist.options import Options

from greedyfhist.utils.metrics import compute_distance_for_lm
from greedyfhist.registration.greedy_f_hist import RegResult, GreedyFHist


def get_core_names():
    return [x.split('.')[0] for x in os.listdir('../../spami/configs/cores_hr/') if x.endswith('.json')]


def compute_tre_with_dfs(warped_df, fixed_df, shape):
    unified_lms = compute_distance_for_lm(warped_df, fixed_df)
    diag = np.sqrt(np.square(shape[0]) + np.square(shape[1]))
    unified_lms['rtre'] = unified_lms['tre']/diag
    mean_rtre = np.mean(unified_lms['rtre'])
    median_rtre = np.median(unified_lms['rtre'])
    median_tre = np.median(unified_lms['tre'])
    mean_tre = np.mean(unified_lms['tre'])
    return mean_rtre, median_rtre, mean_tre, median_tre
    

def update_args(args, core_name, fixed_section, moving_section):
    if core_name == '047_02' and fixed_section == 6 and moving_section == 7:
        print('Updating parameters to fit better.')
        args['mov_sr'] = 15
        args['mov_sp'] = 15
        return args
    else:
        print('Not updating args.')
        return args

def load_transformations(directory):
    """
    Transformations are stored in sub directories in form {target_section}_{source_section}.
    """
    transformations = {}
    for sub_dir in os.listdir(directory):
        sub_dir_ = join(directory, sub_dir, 'out')
        if not os.path.isdir(sub_dir_):
            continue
        transformation = RegResult.from_directory(sub_dir_)
        transformations[sub_dir] = transformation
    return transformations

def load_all_core_transformations(core_name):
    transformations = {}
    section_distances = [1,2,3,4,5]
    for section_distance in section_distances:
        dir_ = f'../save_directories/multisection_integration/direct/multistep_handover_{section_distance}_step_test/{core_name}'
        transformations_ = load_transformations(dir_)
        for key in transformations_:
            transformations[key] = transformations_[key]
    return transformations


def get_all_paths():
    return [
        (1, 6),
        # (1,2,6),
        # (1,3,6),
        (1,2,3,6),
        (2, 7),
        # (2,3,7),
        # (2,6,7),
        (2,3,6,7),
        (3,8),
        # (3,6,8),
        # (3,7,8),
        (3,6,7,8),
        (6,11),
        (6,7,8,9,10,11),
    ]
        
def do_registration_stats(root_target_dir):
    skip_processed_cores = False
    section_distance = 5
    reg_config = {
        'path_to_greedy': '/mnt/work/workbench/maximilw/applications/test/greedy/build2/'
    } 
    registerer = greedy_f_hist.GreedyFHist.load_from_config(reg_config)
    if not os.path.exists(root_target_dir):
        os.mkdir(root_target_dir)
    print(f'Current root dir: {root_target_dir}')
    core_names = get_core_names()
    for core_name in core_names:
        if core_name in ['003_01']:
            continue
        print(f'Working on core: {core_name}')
        target_dir = os.path.join(root_target_dir, core_name)
        if skip_processed_cores and exists(target_dir):
            print(f'Core: {core_name} was already processed.')
            continue
        create_if_not_exists(target_dir)
        print(f'Target dir: {target_dir}')

        config_path = os.path.join('../../spami/configs/cores_hr/', core_name + '.json')
        print(config_path)

        graph = RegGraph.from_config_path(config_path, remove_additional_data=True)    
        # nodes = [x for x in graph.sections if x > 3]
        core_df = pd.DataFrame()
        paths = get_all_paths()
        df = pd.DataFrame()
        for path in paths:
            section_list = [graph.sections[x].copy() for x in path]
            if section_list[0].landmarks is None or section_list[-1].landmarks is None:
                print(f'No landmarks for path: {path}.')
                continue
            section_input_list = [(x.image.data, x.segmentation_mask.data) for x in section_list]
            path_dir = join(target_dir, '_'.join([str(x) for x in path]))
            fixed_section_id = section_list[-1].id_
            moving_section_id = section_list[0].id_
            n_sections = len(section_list)
            options = Options()
            create_if_not_exists(path_dir)
            options.output_directory = path_dir
            registration_result = registerer.register_multi_image(section_input_list, options)
            warped_image_res = registerer.transform_image(section_list[0].image.data, registration_result, 'LINEAR')
            warped_image = warped_image_res.final_transform.registered_image.copy()
            warped_pointcloud = registerer.transform_pointset(section_list[0].landmarks.data, registration_result)
            warped_pointcloud = warped_pointcloud.final_transform.pointcloud.copy()
            warped_pointcloud['label'] = section_list[0].landmarks.data.label
            warped_pointcloud = warped_pointcloud.set_index('label').copy()
            mean_rtre, median_rtre, mean_tre, median_tre = compute_tre_with_dfs(section_list[-1].landmarks.data,
                                                                                warped_pointcloud,
                                                                                section_list[-1].image.data.shape)    
            warped_image[warped_image > 1] = 1
            warped_image = (warped_image * 255).astype(np.uint8)
            sitk.WriteImage(sitk.GetImageFromArray(warped_image), join(path_dir, 'warped_image.png'))
            warped_pointcloud.to_csv(join(path_dir, 'warped_pointcloud.csv'))

            row = {
                'core_name': core_name,
                'fixed_section_id': fixed_section_id,
                'moving_section_id': moving_section_id,
                'mean_rtre': mean_rtre,
                'median_rtre': median_rtre,
                'mean_tre': mean_tre,
                'median_tre': median_tre,
                'path': ','.join([str(x) for x in path]),
                'n_sections': n_sections
            }
            print(row)
            row = pd.DataFrame(row, index=[0])
            df = pd.concat([df, row]).reset_index(drop=True)
        df.to_csv(join(target_dir, 'stats.csv'),index=False)
        core_df = pd.concat([core_df, df]).reset_index(drop=True)
    core_df.to_csv(join(root_target_dir, 'stats.csv'), index=False)
            
            

def main():
    # section_distances = [3,4,5]
    #root_dir = '../save_directories/multisection_integration/direct/multiomics_registration_exhaustive/'
    root_dir = '/mnt/scratch/maximilw/multiomics_integration/multiomics_registration_exhaustive_final'
    create_if_not_exists(root_dir)
    do_registration_stats(root_dir)
        
if __name__ == '__main__':
    main()
