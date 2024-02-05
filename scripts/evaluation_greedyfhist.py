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

        

def get_core_names():
    return [x.split('.')[0] for x in os.listdir('configs/cores_hr/')]


def get_default_args():
    return {
        'resolution': 1024,
        'use_segmentation_masks': True,    
        'output_dir': 'save_directories/temp_pairwise/',
        'tmp_dir': 'save_directories/temp_pairwise/tmp',
        'cleanup_temporary_directories': True,
        'remove_temp_directory': False,
        'cost_fun': 'WNCC',
        'ia': 'ia-com-init',
        'affine_use_denoising': True,
        'deformable_use_denoising': True
}


def main():
    skip_processed_cores = True
    save_plots = False
    root_dir = 'save_directories/pairwise_analysis/greedy_f_hist'
    plot_root_dir = 'save_directories/pairwise_analysis/greedy_f_hist_plots'
    create_if_not_exists(plot_root_dir)
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    core_names = get_core_names()
    df_all = pd.DataFrame()
    for core_name in core_names:
        print(f'Working on core: {core_name}')       
        target_dir = join(root_dir, core_name)
        if os.path.exists(target_dir) and skip_processed_cores:
            print('Already processed!')
            continue
        create_if_not_exists(target_dir)
        config_path = os.path.join('configs/cores_hr/', core_name + '.json')
        
        with open(config_path, 'r') as f:
            config = json.load(f)
            config = clean_configs(config)
            # config = filter_node_ids(config, section_ids)
        graph = RegGraph.from_config(config)        
        df_core = pd.DataFrame()
        section_ids = list(graph.sections)
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
            warp_dir = join(sub_dir, 'warped_section')
            create_if_not_exists(warp_dir)
            args = get_default_args()
            registration_result = graph.default_registerer.coregister_images(moving_img=source_section.image.data,
                                                                             fixed_img=target_section.image.data,
                                                                             moving_img_mask=source_section.segmentation_mask.data,
                                                                             fixed_img_mask=target_section.segmentation_mask.data,
                                                                             args=args)  
            warped_section = source_section.warp(graph.default_registerer, registration_result, args) # Warp section here
            warped_section.store(warp_dir)
            mean_rtre, median_rtre, mean_tre, median_tre = compute_tre(target_section, warped_section)
            row = {
                'core_name': core_name,
                'fixed_section_id': target_idx,
                'moving_section_id': source_idx,
                'mean_rtre': mean_rtre,
                'median_rtre': median_rtre,
                'mean_tre': mean_tre,
                'median_tre': median_tre
            }
            print(row)
            row = pd.DataFrame(row, index=[0])
            df_core = pd.concat([df_core, row]).reset_index(drop=True)
            if save_plots:
                save_dir = join(plot_root_dir, core_name)
                create_if_not_exists(save_dir)
                save_path = join(save_dir, f'{source_idx}_{target_idx}.png')
                plot_registration_summary(source_section, target_section, warped_section, save_path, with_landmarks=True)
                                 
        df_core.to_csv(join(target_dir, 'stats.csv'))
        df_all = pd.concat([df_all, df_core]).reset_index(drop=True)
        
    df_all.to_csv(join(root_dir, 'stats.csv'))
                    

if __name__ == '__main__':
    main()
