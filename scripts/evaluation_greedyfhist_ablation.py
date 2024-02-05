"""
Pairwise registration evaluation with meanshift filter denoising for both affine and deformable registration.
"""
import sys
sys.path.append('..')
sys.path.append('../../spami/')

import json
import os
from os.path import join, exists

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spami.reg_graph import RegGraph
from spami.utils.utils import create_if_not_exists, clean_configs
from spami.utils.plot import plot_registration_summary
from spami.utils.image_utils import get_symmetric_padding_for_sections
from spami.utils.metrics import compute_tre

from greedyfhist.registration.greedy_f_hist import RegResult, GreedyFHist, get_default_args
from greedyfhist.registration import greedy_f_hist
import greedyfhist.registration.greedy_f_hist_old_preprocessing as g_old

# from GreedyFHist.utils.metrics import compute_tre

        

def get_core_names():
    return [x.split('.')[0] for x in os.listdir('../../spami/configs/cores_hr/')]


def get_default_args():
    return {
        'kernel_divider': 100,
        'resolution': 1024,
        'use_segmentation_masks': True,    
        'output_dir': 'save_directories/temp_pairwise/',
        'tmp_dir': 'save_directories/temp_pairwise/tmp',
        'cleanup_temporary_directories': False,
        'remove_temp_directory': False,
        'cost_fun': 'WNCC',
        'ia': 'ia-com-init',
        'affine_use_denoising': True,
        'deformable_use_denoising': True
}

def get_args_no_seg_no_denoising():
    args = greedy_f_hist.get_default_args()
    args['affine_use_denoising'] = False
    args['deformable_use_denoising'] = False
    args['ia'] = 'ia-image-centers'
    return args
    
def get_args_no_denoising():
    args = greedy_f_hist.get_default_args()
    args['affine_use_denoising'] = False
    args['deformable_use_denoising'] = False
    return args

def get_args_no_seg():
    args = greedy_f_hist.get_default_args()
    args['ia'] = 'ia-image-centers'
    return args
    

def register_and_warp(registerer, source_section, target_section, args, source_mask=None, target_mask=None):
    if source_mask is None:
        source_mask = source_section.segmentation_mask.data
    if target_mask is None:
        target_mask = target_section.segmentation_mask.data
    registration_result = registerer.register(moving_img=source_section.image.data,
                                                       fixed_img=target_section.image.data,
                                                       moving_img_mask=source_mask,
                                                       fixed_img_mask=target_mask,
                                                       args=args)  
    warped_section = source_section.warp(registerer, registration_result, args) # Warp section here
            # warped_section.store(warp_dir)
    mean_rtre, median_rtre, mean_tre, median_tre = compute_tre(target_section, warped_section, target_section.image.data.shape)    
    return mean_rtre, median_rtre, mean_tre, median_tre, warped_section

def ablation_case1(registerer, source_section, target_section, core_name, source_idx, target_idx):
    args = get_args_no_seg_no_denoising()
    src_width, src_height, _ = source_section.image.data.shape
    dst_width, dst_height, _ = target_section.image.data.shape
    source_mask = np.ones((src_width, src_height), dtype=np.uint8)
    target_mask = np.ones((dst_width, dst_height), dtype=np.uint8)

    registerer = g_old.
    mean_rtre, median_rtre, mean_tre, median_tre, warped_section = register_and_warp(registerer, source_section, target_section, args, source_mask=source_mask, target_mask=target_mask)
    row = {
        'core_name': core_name,
        'ablation_variant': 'no_seg_no_denoising',
        'fixed_section_id': target_idx,
        'moving_section_id': source_idx,
        'mean_rtre': mean_rtre,
        'median_rtre': median_rtre,
        'mean_tre': mean_tre,
        'median_tre': median_tre
    }
    print(row)
    row = pd.DataFrame(row, index=[0])
    return row, warped_section
    
def ablation_case2(registerer, source_section, target_section, core_name, source_idx, target_idx):
    args = get_args_no_denoising()   
    mean_rtre, median_rtre, mean_tre, median_tre, warped_section = register_and_warp(registerer, source_section, target_section, args)
    row = {
        'core_name': core_name,
        'ablation_variant': 'no_denoising',
        'fixed_section_id': target_idx,
        'moving_section_id': source_idx,
        'mean_rtre': mean_rtre,
        'median_rtre': median_rtre,
        'mean_tre': mean_tre,
        'median_tre': median_tre
    }
    print(row)
    row = pd.DataFrame(row, index=[0])
    return row, warped_section
    
def ablation_case3(registerer, source_section, target_section, core_name, source_idx, target_idx):
    args = get_args_no_seg()
    src_width, src_height, _ = source_section.image.data.shape
    dst_width, dst_height, _ = target_section.image.data.shape
    source_mask = np.ones((src_width, src_height), dtype=np.uint8)
    target_mask = np.ones((dst_width, dst_height), dtype=np.uint8)

    mean_rtre, median_rtre, mean_tre, median_tre, warped_section = register_and_warp(registerer, source_section, target_section, args, source_mask=source_mask, target_mask=target_mask)
    row = {
        'core_name': core_name,
        'ablation_variant': 'no_seg',
        'fixed_section_id': target_idx,
        'moving_section_id': source_idx,
        'mean_rtre': mean_rtre,
        'median_rtre': median_rtre,
        'mean_tre': mean_tre,
        'median_tre': median_tre
    }
    print(row)
    row = pd.DataFrame(row, index=[0])
    return row, warped_section    

def ablation_case4(registerer, source_section, target_section, core_name, source_idx, target_idx):
    args = greedy_f_hist.get_default_args()
    mean_rtre, median_rtre, mean_tre, median_tre, warped_section = register_and_warp(registerer, source_section, target_section, args)
    row = {
        'core_name': core_name,
        'ablation_variant': 'standard',
        'fixed_section_id': target_idx,
        'moving_section_id': source_idx,
        'mean_rtre': mean_rtre,
        'median_rtre': median_rtre,
        'mean_tre': mean_tre,
        'median_tre': median_tre
    }
    print(row)
    row = pd.DataFrame(row, index=[0])
    return row, warped_section      
    
def main():
    skip_processed_cores = True
    save_plots = False
    root_dir = '../save_directories/pairwise_analysis/greedy_f_hist_ablation'
    create_if_not_exists(root_dir)
    reg_config = {
        'path_to_greedy': '/mnt/work/workbench/maximilw/applications/test/greedy/build2/'
    } 
    registerer = greedy_f_hist.GreedyFHist.load_from_config(reg_config)
    core_names = get_core_names()
    df_all = pd.DataFrame()
    for core_name in core_names:
        print(f'Working on core: {core_name}')       
        target_dir = join(root_dir, core_name)
        if os.path.exists(target_dir) and skip_processed_cores:
            print('Already processed!')
            continue
        create_if_not_exists(target_dir)
        config_path = os.path.join('../../spami/configs/cores_hr/', core_name + '.json')
        
        with open(config_path, 'r') as f:
            config = json.load(f)
            config = clean_configs(config)
            # config = filter_node_ids(config, section_ids)
        graph = RegGraph.from_config(config)        
        df_core = pd.DataFrame()
        section_ids = sorted(list(graph.sections))
        # paddings = get_symmetric_padding_for_sections([graph.sections[x] for x in section_ids])
        
        for i in range(len(section_ids)-1):
            source_idx = section_ids[i]
            target_idx = section_ids[i+1]
            source_section = graph.sections[source_idx].copy()
            target_section = graph.sections[target_idx].copy()
            if source_section.landmarks is None or target_section.landmarks is None:
                continue
            print(f'Now processing: {source_idx} and {target_idx}')
            sub_dir = join(target_dir, f'{source_idx}_{target_idx}')
            create_if_not_exists(sub_dir)
            case1_row, case1_warped_section = ablation_case1(registerer, source_section, target_section, core_name, source_idx, target_idx)
            df_core = pd.concat([df_core, case1_row]).reset_index(drop=True)
            case2_row, case2_warped_section = ablation_case2(registerer, source_section, target_section, core_name, source_idx, target_idx)
            df_core = pd.concat([df_core, case2_row]).reset_index(drop=True)
            case3_row, case3_warped_section = ablation_case3(registerer, source_section, target_section, core_name, source_idx, target_idx)
            df_core = pd.concat([df_core, case3_row]).reset_index(drop=True)
            case4_row, case4_warped_section = ablation_case4(registerer, source_section, target_section, core_name, source_idx, target_idx)
            df_core = pd.concat([df_core, case4_row]).reset_index(drop=True)                               
        df_core.to_csv(join(target_dir, 'stats.csv'))
        df_all = pd.concat([df_all, df_core]).reset_index(drop=True)
        
    df_all.to_csv(join(root_dir, 'stats.csv'))
                    

if __name__ == '__main__':
    main()
