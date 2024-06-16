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


def main():
    skip_processed_cores = True
    root_dir = '../save_directories/pairwise_analysis/greedy_f_hist_params_sigma_test2'
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    core_names = get_core_names()
    df_all = pd.DataFrame()
    # resolutions = [1024]
    registerer = GreedyFHist(path_to_greedy='/mnt/work/workbench/maximilw/applications/test/greedy/build2/greedy')
    core_name = '048_01'
    config_path = os.path.join('../../spami/configs/cores_hr/', core_name + '.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
        config = clean_configs(config)
        # config = filter_node_ids(config, section_ids)
    graph = RegGraph.from_config(config)        
    source_section = graph.sections[2].copy()
    target_section = graph.sections[3].copy()
    sigma1s = np.linspace(10, 150, 15)
    sigma2s = np.linspace(10, 150, 15)
    df = pd.DataFrame()
    for s1 in sigma1s:
        for s2 in sigma2s:
            options = Options()
            options.greedy_opts.n_threads = 32
            options.greedy_opts.s1 = s1
            options.greedy_opts.s2 = s2
            start = time.time()
            output_dir = 'save_directories/temp_nb/'
            temp_dir = 'save_directories/temp_nb/temp'
            options.output_directory = output_dir
            options.temporary_directory = temp_dir
            # args['output_dir'] = output_dir

            if exists(options.output_directory):
                shutil.rmtree(options.output_directory)
            create_if_not_exists(options.output_directory)
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
                'fixed_section_id': 3,
                'moving_section_id': 2,
                'mean_rtre': mean_rtre,
                'median_rtre': median_rtre,
                'mean_tre': mean_tre,
                'median_tre': median_tre,
                'duration': duration,
                's1': s1,
                's2': s2,
            }
            row = pd.DataFrame(row, index=[0])
            df = pd.concat([df, row]).reset_index(drop=True)
            print(row)
            print(f'Duration: {duration}.')
    df.to_csv(join(root_dir, 'stats.csv'))


if __name__ == '__main__':
    main()
