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


# def get_default_args():
#     return {
#         'resolution': 1024,
#         'use_segmentation_masks': True,    
#         'output_dir': '../save_directories/temp_pairwise/',
#         'tmp_dir': '../save_directories/temp_pairwise/tmp',
#         'cleanup_temporary_directories': False,
#         'remove_temp_directory': False,
#         'cost_fun': 'WNCC',
#         'ia': 'ia-com-init',
#         'affine_use_denoising': True,
#         'deformable_use_denoising': True
# }

reg_configs = [
    # {'resolution': (128, 128), 'kernel': 2},
    # {'resolution': (256, 256), 'kernel': 3},
    # {'resolution': (512, 512), 'kernel': 5},
    {'resolution': (1024, 1024), 'kernel': 10},
    # {'resolution': (2048, 2048), 'kernel': 20}
]

def main():
    skip_processed_cores = True
    root_dir = '../save_directories/pairwise_analysis/greedy_f_hist_params_no_def_denoising'
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    core_names = get_core_names()
    df_all = pd.DataFrame()
    # resolutions = [1024]
    registerer = GreedyFHist(path_to_greedy='/mnt/work/workbench/maximilw/applications/test/greedy/build2/greedy')

    for reg_config in reg_configs:
    # for resolution in resolutions:
        resolution = reg_config['resolution'][0]
        res_target_dir = f'{root_dir}/{resolution}'
        create_if_not_exists(res_target_dir)
        for core_name in core_names:
            print(f'Working resolution: {resolution} and core: {core_name}')       
            target_dir = join(res_target_dir, core_name)
            # if os.path.exists(target_dir) and skip_processed_cores:
            #     print('Already processed!')
            #     continue
            create_if_not_exists(target_dir)
            config_path = os.path.join('../../spami/configs/cores_hr/', core_name + '.json')
            with open(config_path, 'r') as f:
                config = json.load(f)
                config = clean_configs(config)
                # config = filter_node_ids(config, section_ids)
            graph = RegGraph.from_config(config)        
            df_core = pd.DataFrame()
            section_ids = list(graph.sections)
            # paddings = get_symmetric_padding_for_sections([graph.sections[x] for x in section_ids])

            for i in range(len(section_ids)-1):
                # Add a timer for sleeping, since sometimes greedyfhist get weird results 
                time.sleep(10)
                source_idx = section_ids[i+1]
                target_idx = section_ids[i]
                source_section = graph.sections[source_idx].copy()
                target_section = graph.sections[target_idx].copy()
                if source_section.landmarks is None or target_section.landmarks is None:
                    continue
                print(f'Now processing: {source_idx} and {target_idx}')
                sub_dir = join(target_dir, f'{source_idx}_{target_idx}')
                create_if_not_exists(sub_dir)
                # args = get_default_args()
                options = Options()
                options.resolution = reg_config['resolution']
                options.greedy_opts.kernel_size = reg_config['kernel']
                # options.deformable_do_denoising = False
                options.greedy_opts.n_threads = 32
                # args = get_default_args()
                # args['resolution'] = reg_config['resolution']
                # args['kernel'] = reg_config['kernel']
                # args['deformable_use_denoising'] = False
                options.deformable_do_denoising=False
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
                src_landmarks = source_section.landmarks.data[['x', 'y']].to_numpy()
                warped_pointset = registerer.transform_pointset(src_landmarks, registration_result.moving_transform)
                warped_pointset_df = pd.DataFrame(warped_pointset, columns=['x', 'y'])
                warped_pointset_df['label'] = source_section.landmarks.data.label.copy()
                # transformation_result = registerer.transform_pointset(source_section.landmarks.data, registration_result)
                # warped_pointset = transformation_result.final_transform.pointcloud
                # warped_pointset['label'] = source_section.landmarks.data.label
                target_pointset = target_section.landmarks.data
                shape = target_section.image.data.shape
                mean_rtre, median_rtre, mean_tre, median_tre = compute_tre(target_pointset, warped_pointset_df, shape)            
                duration = end - start
                # mean_rtre, median_rtre, mean_tre, median_tre = compute_tre_sections_(target_section, warped_section)
                row = {
                    'core_name': core_name,
                    'fixed_section_id': target_idx,
                    'moving_section_id': source_idx,
                    'mean_rtre': mean_rtre,
                    'median_rtre': median_rtre,
                    'mean_tre': mean_tre,
                    'median_tre': median_tre,
                    'duration': duration,
                    'resolution': resolution
                }
                print(row)
                print(f'Duration: {duration}.')
                row = pd.DataFrame(row, index=[0])
                df_core = pd.concat([df_core, row]).reset_index(drop=True)
            df_core.to_csv(join(target_dir, 'stats.csv'))
            df_all = pd.concat([df_all, df_core]).reset_index(drop=True)
        df_all.to_csv(join(root_dir, 'stats.csv'))
                    

if __name__ == '__main__':
    main()
