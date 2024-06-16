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
    
direct_pairs = [
    (1, 6),
    (2, 7),
    (3, 8),
    (6, 11),
]

grouwpise_reg_chains = [
    (1, 2, 3, 6),
    (2, 3, 6, 7),
    (3, 6, 7, 8),
    (6, 7, 8, 9, 10, 11),
]


def get_sections(sections, idxs):
    section_list = []
    for idx in idxs:
        if idx in sections:
            section_list.append(sections[idx])
    return section_list
    

def main():
    skip_processed_cores = False
    # root_dir = '../save_directories/pairwise_analysis/greedy_f_hist_params_sigma_test_all_cores'
    # root_dir = '../save_directories/pairwise_analysis/greedy_f_hist_params_no_msf_def'
    root_dir = '../save_directories/pairwise_analysis/greedy_f_hist_groupwise_direct'
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
        if core_name != '004_02':
            continue
        sections = load_core(join(src_dir, core_name), skip_section=[4])
        df = pd.DataFrame()
        target_dir = join(root_dir, core_name)
        if exists(target_dir) and skip_processed_cores:
            continue
        create_if_not_exists(target_dir)
        lm_dir = join(target_dir, 'landmarks')
        create_if_not_exists(lm_dir)
        for idx, pair in enumerate(direct_pairs):
            source_idx, target_idx = pair
            if source_idx not in sections or target_idx not in sections:
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
                options.greedy_opts.s1 = 5
                options.greedy_opts.s2 = 5
                options.deformable_do_denoising = False
                options.temporary_directory = 'groupwise_test'
                create_if_not_exists(options.temporary_directory)
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

def main2():
    skip_processed_cores = True
    # root_dir = '../save_directories/pairwise_analysis/greedy_f_hist_params_sigma_test_all_cores'
    # root_dir = '../save_directories/pairwise_analysis/greedy_f_hist_params_no_msf_def'
    root_dir = '../save_directories/pairwise_analysis/greedy_f_hist_groupwise_normal'
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    # core_names = get_core_names()
    df_all = pd.DataFrame()
    # resolutions = [1024]
    # registerer = GreedyFHist(path_to_greedy='/mnt/work/workbench/maximilw/applications/test/greedy/build2/greedy')
    registerer = GreedyFHist.load_from_config({})
    src_dir = '/mnt/scratch/maximilw/data/spatial_multiomics_sections'
    core_names = os.listdir(src_dir)
    for core_name in core_names:
        sections = load_core(join(src_dir, core_name), skip_section=[4])
        df = pd.DataFrame()
        target_dir = join(root_dir, core_name)
        if exists(target_dir) and skip_processed_cores:
            continue
        create_if_not_exists(target_dir)
        lm_dir = join(target_dir, 'landmarks')
        create_if_not_exists(lm_dir)
        create_if_not_exists(target_dir)
        for idx, reg_chain in enumerate(grouwpise_reg_chains):
            section_list = get_sections(sections, reg_chain)
            source_idx = reg_chain[0]
            target_idx = reg_chain[-1]
            source_section = section_list[source_idx]
            target_section = section_list[target_idx]
            if source_section.landmarks is None or target_section.landmarks is None:
                continue
            print(f'Now processing: {source_idx} and {target_idx}')              
            img_mask_list = []
            for section in section_list:
                img = section.image.data
                # mask = section.get_annotations_by_names('tissue_mask').data if section.get_annotations_by_names('tissue_mask') is not None else None
                mask = None
                img_mask_list.append((img, mask))
            time.sleep(5)
            options = Options()
            options.greedy_opts.n_threads = 32
            options.greedy_opts.s1 = 5
            options.greedy_opts.s2 = 5
            options.deformable_do_denoising = False
            if exists(options.output_directory):
                shutil.rmtree(options.output_directory)
            create_if_not_exists(options.output_directory)
            start = time.time()                
            start = time.time()                
            transforms, group_reg = registerer.groupwise_registration(img_mask_list, nonrigid_option=options)
            end = time.time()
            reg_result = transforms[0]
            lms = source_section.landmarks.data[['x', 'y']].to_numpy()
            warped_pointset = registerer.transform_pointset(lms, reg_result.moving_transform)
            warped_pointset_df = pd.DataFrame(warped_pointset, columns=['x', 'y'])
            warped_pointset_df['label'] = source_section.landmarks.data.label
            target_pointset = target_section.landmarks.data
            shape = target_section.image.data.shape
            mean_rtre, median_rtre, mean_tre, median_tre = compute_tre(target_pointset, warped_pointset_df, shape)            
            lm_subdir = join(lm_dir, f'{source_idx}_{target_idx}')
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
                'duration': duration
            }
            row = pd.DataFrame(row, index=[0])
            df = pd.concat([df, row]).reset_index(drop=True)
            print(row)
            print(f'Duration: {duration}.')
        df.to_csv(join(target_dir, 'stats.csv'))
        df_all = pd.concat([df_all, df]).reset_index(drop=True)
    df_all.to_csv(join(root_dir, 'stats.csv'))

def main3():
    skip_processed_cores = False
    # root_dir = '../save_directories/pairwise_analysis/greedy_f_hist_params_sigma_test_all_cores'
    # root_dir = '../save_directories/pairwise_analysis/greedy_f_hist_params_no_msf_def'
    root_dir = '/mnt/scratch/maximilw/data/groupwise_evaluation/test1'
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    # core_names = get_core_names()
    # resolutions = [1024]
    # registerer = GreedyFHist(path_to_greedy='/mnt/work/workbench/maximilw/applications/test/greedy/build2/greedy')
    registerer = GreedyFHist.load_from_config({})
    src_dir = '/mnt/scratch/maximilw/data/spatial_multiomics_sections'
    core_names = os.listdir(src_dir)
    for core_name in core_names:
        print(core_name)
        sections = load_core(join(src_dir, core_name), skip_section=[4])
        target_dir = join(root_dir, core_name)
        if exists(target_dir) and skip_processed_cores:
            continue
        create_if_not_exists(target_dir)
        section_ids = sorted(sections.keys())
        section_ids = [x for x in section_ids if x != 4]
        img_mask_list = []
        for section_id in section_ids:
            img_mask_list.append((sections[section_id].image.data, None))
        # for idx, reg_chain in enumerate(grouwpise_reg_chains):
        #     section_list = get_sections(sections, reg_chain)
        #     source_idx = reg_chain[0]
        #     target_idx = reg_chain[-1]
        #     source_section = section_list[source_idx]
        #     target_section = section_list[target_idx]
        #     if source_section.landmarks is None or target_section.landmarks is None:
        #         continue
        #     print(f'Now processing: {source_idx} and {target_idx}')              
        #     img_mask_list = []
        #     for section in section_list:
        #         img = section.image.data
        #         # mask = section.get_annotations_by_names('tissue_mask').data if section.get_annotations_by_names('tissue_mask') is not None else None
        #         mask = None
        #         img_mask_list.append((img, mask))
        # options = Options()
        # options.greedy_opts.n_threads = 32
        # options.greedy_opts.s1 = 5
        # options.greedy_opts.s2 = 5
        # options.deformable_do_denoising = False
        # if exists(options.output_directory):
        #     shutil.rmtree(options.output_directory)
        # create_if_not_exists(options.output_directory)
        time.sleep(5)
        start = time.time()                
        transforms, warped_aff_images = registerer.groupwise_registration(img_mask_list, skip_deformable_registration=True) 
        end = time.time()
        for idx, aff_transform in enumerate(transforms.affine_transform):
            source_id = section_ids[idx]
            target_id = section_ids[idx+1]
            dir_name = f'{source_id}_{target_id}'
            aff_dir = join(target_dir, dir_name)
            create_if_not_exists(aff_dir)
            aff_transform.to_file(aff_dir)
        duration = end - start
        print(f'Duration: {duration}.')
        # reg_result = transforms[0]
        # lms = source_section.landmarks.data[['x', 'y']].to_numpy()
        # warped_pointset = registerer.transform_pointset(lms, reg_result.moving_transform)
        # warped_pointset_df = pd.DataFrame(warped_pointset, columns=['x', 'y'])
        # warped_pointset_df['label'] = source_section.landmarks.data.label
        # target_pointset = target_section.landmarks.data
        # shape = target_section.image.data.shape
        # mean_rtre, median_rtre, mean_tre, median_tre = compute_tre(target_pointset, warped_pointset_df, shape)            
        # lm_subdir = join(lm_dir, f'{source_idx}_{target_idx}')
        # create_if_not_exists(lm_subdir)
        # target_pointset_path = join(lm_subdir, f'{target_idx}_lms.csv')
        # warped_pointset_path = join(lm_subdir, f'{source_idx}_lms_warped.csv')
        # target_pointset.to_csv(target_pointset_path)
        # warped_pointset_df.to_csv(warped_pointset_path)
        # duration = end - start
        # row = {
        #     'core_name': core_name,
        #     'fixed_section_id': target_idx,
        #     'moving_section_id': source_idx,
        #     'mean_rtre': mean_rtre,
        #     'median_rtre': median_rtre,
        #     'mean_tre': mean_tre,
        #     'median_tre': median_tre,
        #     'duration': duration
        # }
        # row = pd.DataFrame(row, index=[0])
        # df = pd.concat([df, row]).reset_index(drop=True)
        # print(row)
        # print(f'Duration: {duration}.')
    #     df.to_csv(join(target_dir, 'stats.csv'))
    #     df_all = pd.concat([df_all, df]).reset_index(drop=True)
    # df_all.to_csv(join(root_dir, 'stats.csv'))


if __name__ == '__main__':
    main()
    # main2()
    # main3()