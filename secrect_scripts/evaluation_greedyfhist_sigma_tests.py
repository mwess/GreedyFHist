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


def eucl(src, dst):
    return np.sqrt(np.square(src[:, 0] - dst[:, 0]) + np.square(src[:, 1] - dst[:, 1]))


def get_core_names():
    return [x.split('.')[0] for x in os.listdir('../../spami/configs/cores_hr/')]


def main():
    skip_processed_cores = True
    root_dir = '../save_directories/pairwise_analysis/greedy_f_hist_params_sigma_test4'
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    core_names = get_core_names()
    # df_all = pd.DataFrame()
    # resolutions = [1024]
    # registerer = GreedyFHist(path_to_greedy='/mnt/work/workbench/maximilw/applications/test/greedy/build2/greedy')
    # core_name = '048_01'
    # config_path = os.path.join('../../spami/configs/cores_hr/', core_name + '.json')
    # with open(config_path, 'r') as f:
    #     config = json.load(f)
    #     config = clean_configs(config)
    #     # config = filter_node_ids(config, section_ids)
    # graph = RegGraph.from_config(config)        
    # source_section = graph.sections[1].copy()
    # target_section = graph.sections[2].copy()
    # sigma1s = np.linspace(0, 10, 11)
    # sigma2s = np.linspace(0, 10, 11)
    sigma1s = [6]
    sigma2s = [5]
    for s1 in sigma1s:
        for s2 in sigma2s:
            s1 = int(s1)
            s2 = int(s2)
            # if s1 != 2 and s2 != 1:
            #     continue
            sigma_dir = join(root_dir, f'{s1}_{s2}')
            create_if_not_exists(sigma_dir)
            for core_name in core_names:
                #if core_name == '048_01':
                #    continue
                target_dir = join(sigma_dir, core_name)
                registerer = GreedyFHist(path_to_greedy='/mnt/work/workbench/maximilw/applications/test/greedy/build2/greedy')
                # core_name = '048_01'
                config_path = os.path.join('../../spami/configs/cores_hr/', core_name + '.json')
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    config = clean_configs(config)
                    # config = filter_node_ids(config, section_ids)
                graph = RegGraph.from_config(config)     
                section_ids = list(graph.sections)
                # target_dir = join(root_dir, core_name)
                create_if_not_exists(target_dir)
                df = pd.DataFrame()
                for i in range(len(section_ids) -1):
                    source_idx = section_ids[i]
                    target_idx = section_ids[i+1]
                    if np.abs(source_idx - target_idx) > 1:
                        continue
                    source_section = graph.sections[source_idx].copy()
                    target_section = graph.sections[target_idx].copy()  
                    if source_section.landmarks is None or target_section.landmarks is None:
                        continue
                    # source_section = graph.sections[1].copy()
                    # target_section = graph.sections[2].copy()        
                    options = Options()
                    options.greedy_opts.n_threads = 32
                    options.greedy_opts.s1 = s1
                    options.greedy_opts.s2 = s2
                    options.deformable_do_denoising=True
                    options.affine_do_denoising=True
                    options.ia = 'ia-image-centers'
                    output_dir = 'save_directories/temp_nb/'
                    temp_dir = 'save_directories/temp_nb/temp'
                    if exists(output_dir):
                        shutil.rmtree(output_dir)
                        os.mkdir(output_dir)
                    options.output_directory = output_dir
                    options.temporary_directory = temp_dir
                    options.remove_temporary_directory = False
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
                        'fixed_section_id': target_idx,
                        'moving_section_id': source_idx,
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
                    warped_df = warped_pointset.copy()
                    fixed_df = target_section.landmarks.data.copy()
                    merged_df = warped_df.merge(fixed_df, on='label', suffixes=('_src', '_dst'))
                    merged_df.replace([np.inf, -np.inf], np.nan, inplace=True)
                    merged_df.dropna(inplace=True)
                    src_mat = merged_df[['x_src', 'y_src']].to_numpy()
                    dst_mat = merged_df[['x_dst', 'y_dst']].to_numpy()
                    section_dir = join(target_dir, f'{source_idx}_{target_idx}')
                    create_if_not_exists(section_dir)
                    merged_df['tre'] = eucl(src_mat, dst_mat)
                    merged_df.to_csv(join(section_dir, 'merged_df.csv'))
                df.to_csv(join(target_dir, 'stats.csv'))


if __name__ == '__main__':
    main()
