Strategy 2 """
Direct multisection integration. Uses symmetric padding and requires segmentation masks.
"""
import sys
sys.path.append('../../spami')
sys.path.append('../')

import json
import os
from os.path import join, exists
import shutil
import time

import pandas as pd
import SimpleITK as sitk

from greedyfhist.registration import greedy_f_hist

from spami.reg_graph import RegGraph
from spami.utils.utils import create_if_not_exists, filter_node_ids
from spami.utils.image_utils import get_symmetric_padding_for_sections
from spami.utils.metrics import compute_tre
from spami.utils.plot import plot_registration_summary

def get_core_names():
    return [x.split('.')[0] for x in os.listdir('../../spami/configs/cores_hr/')]

# def get_default_args():
#     return {
#     'kernel': 10,
#     'resolution': (512, 512),
#     'use_segmentation_masks': True,    
#     'output_dir': '../save_directories/multisection_integration/temp_mint/',
#     'tmp_dir': '../save_directories/multisection_integration/temp_mint/tmp',
#     'cleanup_temporary_directories': False,
#     'remove_temp_directory': False,
#     'cost_fun': 'WNCC',
#     'ia': 'ia-com-init',
#     'affine_use_denoising': True,
#     # 'fix_sr': 20,
#     # 'fix_sp': 15,
#     # 'mov_sr': 15,
#     # 'mov_sp': 15       
# }

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

        
def do_registration_stats(root_dir, section_distance=1):
    # Start pseudoscript for multisection registration
    # root_dir = 'save_directories/multisection_integration/direct/denoised_9'
    skip_processed_cores = True
    save_figs = True

    reg_config = {
        'path_to_greedy': '/mnt/work/workbench/maximilw/applications/test/greedy/build2/'
    } 
    registerer = greedy_f_hist.GreedyFHist.load_from_config(reg_config)
    
    
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    core_names = get_core_names()
    for core_name in core_names:
        print(f'Working on core: {core_name}')
        target_dir = os.path.join(root_dir, core_name)
        if skip_processed_cores and exists(target_dir):
            print(f'Core: {core_name} was already processed.')
            continue
        create_if_not_exists(target_dir)
        print(target_dir)

        config_path = os.path.join('../../spami/configs/cores_hr/', core_name + '.json')
        print(config_path)
        
        graph = RegGraph.from_config_path(config_path, remove_additional_data=True)         
        section_ids = list(graph.sections.keys())
        df = pd.DataFrame()
        for moving_section_id in graph.sections:
            fixed_section_id = moving_section_id + section_distance
            if fixed_section_id not in graph.sections:
                continue
            print(f'Working on moving: {moving_section_id} and {fixed_section_id}.')
            if graph.sections[moving_section_id].landmarks is None or graph.sections[fixed_section_id].landmarks is None:
                print('No landmarks found! Skipping..')
                continue
            path = get_path(moving_section_id, fixed_section_id, section_ids)                
            warped_section, registration_chain = register_along_path(graph, registerer, path)
            sub_dir = join(target_dir, f'{fixed_section_id}_{moving_section_id}')
            create_if_not_exists(sub_dir)
            for idx, temp_section in enumerate(registration_chain):
                sitk.WriteImage(sitk.GetImageFromArray(temp_section.image.data), join(sub_dir, f'registered_image_{idx}.png'))
                temp_section.landmarks.data.to_csv(join(sub_dir, f'registered_landmarks_{idx}.csv'), index=False)
            # print(f'Reg time: {end - start}.')
            fixed_section = graph.sections[path[-1]].copy()
            mean_rtre, median_rtre, mean_tre, median_tre = compute_tre(fixed_section,
                                                                       warped_section,
                                                                       fixed_section.image.data.shape)
            sitk.WriteImage(sitk.GetImageFromArray(warped_section.image.data), join(sub_dir, 'final_registered_image.png'))
            warped_section.landmarks.data.to_csv(join(sub_dir, 'final_registered_landmarks.csv'), index=False)
            row = {
                'core_name': core_name,
                'fixed_section_id': fixed_section_id,
                'moving_section_id': moving_section_id,
                'mean_rtre': mean_rtre,
                'median_rtre': median_rtre,
                'mean_tre': mean_tre,
                'median_tre': median_tre
            }
            print(row)
            row = pd.DataFrame(row, index=[0])
            df = pd.concat([df, row]).reset_index(drop=True)
            remove_if_exists(join(sub_dir, 'reg'))
        df.to_csv(join(target_dir, 'stats.csv'), index=False)

def remove_if_exists(path):
    if exists(path):
        shutil.rmtree(path)

def register_along_path(graph, registerer, path, tmp_path=None):
    moving_section_id = path[0]
    moving_section = graph.sections[moving_section_id].copy()
    registration_chain = []
    for fixed_section_id in path[1:]:
        print(fixed_section_id)
        fixed_section = graph.sections[fixed_section_id].copy()
        
        default_args = greedy_f_hist.get_default_args()
        if tmp_path is not None:
            default_args['tmp_dir'] = tmp_path
            default_args['output_dir'] = join(tmp_path, 'out')
        remove_if_exists(default_args['tmp_dir'])
        remove_if_exists(default_args['output_dir'])
        # warped_dir = join(tmp_path, 'warped_section')
        transformation = registerer.register(moving_section.image.data,
                                             fixed_section.image.data,
                                             moving_section.segmentation_mask.data,
                                             fixed_section.segmentation_mask.data,
                                             default_args)
        warped_section = moving_section.warp(registerer, transformation, default_args) # Warp section here     
        registration_chain.append(warped_section.copy())
        moving_section = warped_section.copy()
    return moving_section, registration_chain

def get_path(start, end, nodes):
    path = [start]
    for i in range(start+1, end):
        if i in nodes:
            path.append(i)
    path.append(end)
    return path
        
        
def main():
    section_distances = [2,3,4,5]
    for section_distance in section_distances:
        root_dir = f'../save_directories/multisection_integration/multistep/denoised_10_{section_distance}_multistep_test'
        # root_dir = f'../save_directories/multisection_integration/direct/denoised_10_small_1_step_big_kernel_test3'
        create_if_not_exists(root_dir)
        do_registration_stats(root_dir, section_distance=section_distance)
    
        
if __name__ == '__main__':
    main()
