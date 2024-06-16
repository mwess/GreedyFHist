"""
Pairwise registration evaluation with meanshift filter denoising for both affine and deformable registration.
"""
import sys
sys.path.append('..')
sys.path.append('../../spami/')
import time

import json
import os
from os.path import join, exists
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from greedyfhist.registration import GreedyFHist, get_default_args
from greedyfhist.options import Options
from greedyfhist.utils.metrics import compute_tre



from miit.reg_graph import RegGraph
from miit.utils.utils import create_if_not_exists, clean_configs
# from miit.utils.metrics import compute_tre, compute_tre_sections_

def get_core_names():
    return [x.split('.')[0] for x in os.listdir('../../spami/configs/cores_hr/')]

sigma_configs = [
    {'s1': 6, 's2': 5},
    {'s1': 120, 's2': 10},    
    {'s1': 100, 's2': 120},    
    {'s1': 140, 's2': 130},    
    {'s1': 3.5, 's2': 1},
    {'s1': -2, 's2': -0.5},
]
    
    

def main():
    skip_processed_cores = True
    root_dir = '../save_directories/pairwise_analysis/greedy_f_hist_params_sigma_test_ia_image_centers'
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    core_names = get_core_names()
    df_all = pd.DataFrame()
    # resolutions = [1024]
    registerer = GreedyFHist(path_to_greedy='/mnt/work/workbench/maximilw/applications/test/greedy/build2/greedy')
    for idx, sigma_config in enumerate(sigma_configs):
        target_dir = f'{root_dir}/{idx}'
        create_if_not_exists(target_dir)
        for core_name in core_names:
            config_path = os.path.join('../../spami/configs/cores_hr/', core_name + '.json')
            with open(config_path, 'r') as f:
                config = json.load(f)
                config = clean_configs(config)
                # config = filter_node_ids(config, section_ids)
            graph = RegGraph.from_config(config)        
            df = pd.DataFrame()
            section_ids = list(graph.sections)
            for i in range(len(section_ids) -1):
                source_idx = section_ids[i]
                target_idx = section_ids[i+1]
                source_section = graph.sections[source_idx].copy()
                target_section = graph.sections[target_idx].copy()
                if source_section.landmarks is None or target_section.landmarks is None:
                    continue
                print(f'Now processing: {source_idx} and {target_idx}')              
                print(f'Sigma config: {sigma_config}')
                options = Options()
                options.greedy_opts.n_threads = 32
                output_dir = 'save_directories/temp_nb/'
                temp_dir = 'save_directories/temp_nb/temp'
                options.output_directory = output_dir
                options.temporary_directory = temp_dir
                options.greedy_opts.s1 = sigma_config['s1']
                options.greedy_opts.s2 = sigma_config['s2']
                # options.ia = 'ia-image-center'
                # args['output_dir'] = output_dir
                if exists(options.output_directory):
                    shutil.rmtree(options.output_directory)
                create_if_not_exists(options.output_directory)
                start = time.time()                
                registration_result = registerer.register(moving_img=source_section.image.data,
                                                          fixed_img=target_section.image.data,
                                                          moving_img_mask=source_section.segmentation_mask.data,
                                                          fixed_img_mask=target_section.segmentation_mask.data,
                                                          options=options)
                end = time.time()
                transformation_result = registerer.transform_pointset(source_section.landmarks.data, registration_result)
                warped_pointset = transformation_result.final_transform.pointcloud
                warped_pointset['label'] = source_section.landmarks.data.label
                target_pointset = target_section.landmarks.data
                shape = target_section.image.data.shape
                mean_rtre, median_rtre, mean_tre, median_tre = compute_tre(target_pointset, warped_pointset, shape)            
                duration = end - start
                row = {
                    'core_name': core_name,
                    'fixed_section_id': source_idx,
                    'moving_section_id': target_idx,
                    'mean_rtre': mean_rtre,
                    'median_rtre': median_rtre,
                    'mean_tre': mean_tre,
                    'median_tre': median_tre,
                    'duration': duration,
                    's1': options.greedy_opts.s1,
                    's2': options.greedy_opts.s2,
                }
                row = pd.DataFrame(row, index=[0])
                df = pd.concat([df, row]).reset_index(drop=True)
                print(row)
                print(f'Duration: {duration}.')
        df.to_csv(join(target_dir, 'stats.csv'))
        df_all = pd.concat([df_all, df]).reset_index(drop=True)
    df_all.to_csv(join(root_dir, 'stats.csv'))


if __name__ == '__main__':
    main()
