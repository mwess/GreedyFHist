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



# from miit.reg_graph import RegGraph
from miit.utils.utils import create_if_not_exists, clean_configs
from miit.spatial_data.section import Section
# from miit.utils.metrics import compute_tre, compute_tre_sections_

def get_core_names():
    return [x.split('.')[0] for x in os.listdir('../../spami/configs/cores_hr/')]

sigma_configs = [
    {'s1': 6, 's2': 5},
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
    skip_processed_cores = True
    # root_dir = '../save_directories/pairwise_analysis/greedy_f_hist_params_sigma_test_all_cores'
    # root_dir = '../save_directories/pairwise_analysis/greedy_f_hist_params_no_msf_def'
    root_dir = '../save_directories/pairwise_analysis/greedy_f_hist_params_standard_no_masks_no_def'
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    # core_names = get_core_names()
    df_all = pd.DataFrame()
    # resolutions = [1024]
    # registerer = GreedyFHist(path_to_greedy='/mnt/work/workbench/maximilw/applications/test/greedy/build2/greedy')
    registerer = GreedyFHist.load_from_config({})
    src_dir = '/mnt/scratch/maximilw/data/spatial_multiomics_sections'
    core_names = os.listdir(src_dir)
    N_ROUNDS = 1
    for core_name in core_names:
        sections = load_core(join(src_dir, core_name), skip_section=[4])
        df = pd.DataFrame()
        section_ids = sorted(list(sections.keys()))
        target_dir = join(root_dir, core_name)
        if exists(target_dir) and skip_processed_cores:
            continue
        create_if_not_exists(target_dir)
        lm_dir = join(target_dir, 'landmarks')
        create_if_not_exists(lm_dir)
        for i in range(len(section_ids) -1):
            source_idx = section_ids[i]
            target_idx = section_ids[i+1]
            # This is the registration of the big gap. Skip that.
            if target_idx == 6:
                continue
            source_section = sections[source_idx]
            target_section = sections[target_idx]
            if source_section.landmarks is None or target_section.landmarks is None:
                continue
            print(f'Now processing: {source_idx} and {target_idx}')              
            create_if_not_exists(target_dir)
            for rep in range(N_ROUNDS):
                time.sleep(5)
                options = Options()
                options.greedy_opts.n_threads = 32
                # output_dir = 'save_directories/temp_nb/'
                # temp_dir = 'save_directories/temp_nb/temp'
                # options.output_directory = output_dir
                # options.temporary_directory = temp_dir
                options.greedy_opts.s1 = 8
                options.greedy_opts.s2 = 6
                options.deformable_do_denoising = False
                if exists(options.output_directory):
                    shutil.rmtree(options.output_directory)
                create_if_not_exists(options.output_directory)
                start = time.time()                
                reg_result = registerer.register(moving_img=source_section.image.data,
                                                 fixed_img=target_section.image.data,
                                                 options=options)
                end = time.time()
                lms = source_section.landmarks.data[['x', 'y']].to_numpy()
                warped_pointset = registerer.transform_pointset(lms, reg_result.moving_transform)
                warped_pointset_df = pd.DataFrame(warped_pointset, columns=['x', 'y'])
                warped_pointset_df['label'] = source_section.landmarks.data.label
                target_pointset = target_section.landmarks.data
                shape = target_section.image.data.shape
                mean_rtre, median_rtre, mean_tre, median_tre = compute_tre(target_pointset, warped_pointset_df, shape)            
                lm_subdir = join(lm_dir, f'{source_idx}_{target_idx}_{rep}')
                create_if_not_exists(lm_subdir)
                target_pointset_path = join(lm_subdir, f'{target_idx}_lms.csv')
                warped_pointset_path = join(lm_subdir, f'{source_idx}_lms_warped.csv')
                target_pointset.to_csv(target_pointset_path)
                warped_pointset_df.to_csv(warped_pointset_path)
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
                    'rep': rep
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
