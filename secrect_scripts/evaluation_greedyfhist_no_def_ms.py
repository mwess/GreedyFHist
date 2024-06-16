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

from greedyfhist.registration import GreedyFHist
from greedyfhist.options import Options
from greedyfhist.utils.metrics import compute_tre



from miit.reg_graph import RegGraph
from miit.utils.utils import create_if_not_exists, clean_configs
from miit.spatial_data.section import Section
# from miit.utils.metrics import compute_tre, compute_tre_sections_

def get_core_names():
    return [x.split('.')[0] for x in os.listdir('../../spami/configs/cores_hr/')]

sigma_configs = [
#    {'s1': 3.5, 's2': 1.0},
#    {'s1': -2.0, 's2': -2.0},
#    {'s1': -2.0, 's2': -0.5},
#    {'s1': -0.5, 's2': -1.0},
#    {'s1': -1.5, 's2': -1.0},
    {'s1': 6, 's2': 5},
    # {'s1': 5, 's2': 6},
    # {'s1': 6, 's2': 4},
    # {'s1': 6, 's2': 6},
    # {'s1': 6, 's2': 7},
    # {'s1': 5, 's2': 4},
    # {'s1': 140, 's2': 90},
    # {'s1': 50, 's2': 150},
    # {'s1': 70, 's2': 100},
    # {'s1': 50, 's2': 130},
    # {'s1': 120, 's2': 120},
    # {'s1': 10, 's2': 110},
]
    
def load_core(directory, skip_section=None):
    if skip_section is None:
        skip_section = []
    sections = {}
    for sub_dir in os.listdir(directory):
        id_ = int(sub_dir)
        if id_ in skip_section:
            continue
        section = Section.load(join(directory, sub_dir))
        sections[id_] = section
    return sections
    

def main():
    skip_processed_cores = False
    # root_dir = '../save_directories/pairwise_analysis/greedy_f_hist_params_sigma_test_all_cores'
    # root_dir = '../save_directories/pairwise_analysis/greedy_f_hist_params_no_msf_def'
    root_dir = '../save_directories/pairwise_analysis/greedy_f_hist_params_no_def_ms'
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    # core_names = get_core_names()
    df_all = pd.DataFrame()
    # resolutions = [1024]
    registerer = GreedyFHist(path_to_greedy='/mnt/work/workbench/maximilw/applications/test/greedy/build2/greedy')
    src_dir = '/mnt/scratch/maximilw/data/spatial_multiomics_sections'
    core_names = os.listdir(src_dir)
    for core_name in core_names:
        sections = load_core(join(src_dir, core_name), skip_section=[4])
        df = pd.DataFrame()
        section_ids = sorted(list(sections.keys()))
        target_dir = join(root_dir, core_name)
        if exists(target_dir) and skip_processed_cores:
            continue
        create_if_not_exists(target_dir)
        for i in range(len(section_ids) -1):
            source_idx = section_ids[i]
            target_idx = section_ids[i+1]
            source_section = sections[source_idx].copy()
            target_section = sections[target_idx].copy()
            if source_section.landmarks is None or target_section.landmarks is None:
                continue
            print(f'Now processing: {source_idx} and {target_idx}')              
            create_if_not_exists(target_dir)
            for sigma_config in sigma_configs:
                time.sleep(10)
                print(f'Sigma config: {sigma_config}')
                options = Options()
                options.greedy_opts.n_threads = 32
                output_dir = 'save_directories/temp_nb/'
                temp_dir = 'save_directories/temp_nb/temp'
                options.output_directory = output_dir
                options.temporary_directory = temp_dir
                options.greedy_opts.s1 = sigma_config['s1']
                options.greedy_opts.s2 = sigma_config['s2']
                options.deformable_do_denoising = False
                # args['output_dir'] = output_dir
                if exists(options.output_directory):
                    shutil.rmtree(options.output_directory)
                create_if_not_exists(options.output_directory)
                start = time.time()                
                moving_mask = source_section.get_annotations_by_names('tissue_mask').data
                fixed_mask = target_section.get_annotations_by_names('tissue_mask').data
                reg_result = registerer.register(moving_img=source_section.image.data,
                                                          fixed_img=target_section.image.data,
                                                          moving_img_mask=moving_mask,
                                                          fixed_img_mask=fixed_mask,
                                                          options=options)
                end = time.time()
                lms = source_section.landmarks.data[['x', 'y']].to_numpy()
                warped_pointset = registerer.transform_pointset(lms, reg_result.moving_transform)
                warped_pointset_df = pd.DataFrame(warped_pointset, columns=['x', 'y'])
                warped_pointset_df['label'] = source_section.landmarks.data.label
                target_pointset = target_section.landmarks.data
                shape = target_section.image.data.shape
                mean_rtre, median_rtre, mean_tre, median_tre = compute_tre(target_pointset, warped_pointset_df, shape)            
                duration = end - start
                row = {
                    'core_name': core_name,
                    'fixed_section_id': target_idx,
                    'moving_section_id': source_idx,
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
