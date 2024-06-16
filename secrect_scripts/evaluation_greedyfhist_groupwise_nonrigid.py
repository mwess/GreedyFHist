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
from greedyfhist.segmentation.segmenation import load_yolo_segmentation
from greedyfhist.registration.greedy_f_hist import GFHTransform, RegistrationResult, GroupwiseRegResult, compose_transforms



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
    
def get_transform_chain(transforms, from_, to):
    chain = []
    start_adding = False
    for idx, (mov_idx, fix_idx, transform) in enumerate(transforms):
        if mov_idx == from_:
            start_adding = True
        if start_adding:
            chain.append((mov_idx, fix_idx, transform))
        if fix_idx == to:
            break
    return chain

def load_transforms(directory):
    transforms = {}
    for sub_dir in os.listdir(directory):
        path = join(directory, sub_dir)
        key1, key2 = sub_dir.split('_')
        key1 = int(key1)
        key2 = int(key2)
        # print(sub_dir)
        transform = RegistrationResult.load(path)
        transforms[key1] = (key1, key2, transform)
    t_ids = sorted(transforms.keys())
    transforms_sorted = []
    for t_id in t_ids:
        # print(t_id)
        transforms_sorted.append(transforms[t_id])
    return transforms_sorted
    

def main4():
    skip_processed_cores = False
    segmentation_function = load_yolo_segmentation()
    # root_dir = '../save_directories/pairwise_analysis/greedy_f_hist_params_sigma_test_all_cores'
    # root_dir = '../save_directories/pairwise_analysis/greedy_f_hist_params_no_msf_def'
    aff_src_dir = '/mnt/scratch/maximilw/data/groupwise_evaluation/test1'
    root_dir = '/mnt/scratch/maximilw/data/groupwise_evaluation/test2'
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    # core_names = get_core_names()
    # resolutions = [1024]
    # registerer = GreedyFHist(path_to_greedy='/mnt/work/workbench/maximilw/applications/test/greedy/build2/greedy')
    registerer = GreedyFHist.load_from_config({})
    src_dir = '/mnt/scratch/maximilw/data/spatial_multiomics_sections'
    df_all = pd.DataFrame()
    core_names = os.listdir(src_dir)
    for core_name in core_names:
        print(core_name)
        sections = load_core(join(src_dir, core_name), skip_section=[4])
        target_dir = join(root_dir, core_name)
        if exists(target_dir) and skip_processed_cores:
            continue
        create_if_not_exists(target_dir)
        lm_dir = join(target_dir, 'landmarks')
        create_if_not_exists(lm_dir)
        df = pd.DataFrame()
        aff_trans_dir = join(aff_src_dir, core_name)
        affine_transforms = load_transforms(aff_trans_dir)
        section_ids = sorted(sections.keys())
        section_ids = [x for x in section_ids if x != 4]
        for (source_idx, target_idx) in direct_pairs:
            source_section = sections[source_idx]
            target_section = sections[target_idx]
            if source_section.landmarks is None or target_section.landmarks is None:
                continue
            moving_image = source_section.image.data
            moving_mask = segmentation_function(moving_image)
            fixed_image = target_section.image.data
            # Load and composite affine transforms manually
            aff_trans_chain = get_transform_chain(affine_transforms, source_idx, target_idx)
            # Remove indices
            aff_trans_chain = [x[2] for x in aff_trans_chain]
            composited_fixed_transform = compose_transforms([x.fixed_transform for x in aff_trans_chain])
            composited_moving_transform = compose_transforms([x.moving_transform for x in aff_trans_chain][::-1])
            reg_result = RegistrationResult(composited_fixed_transform, composited_moving_transform)
            aff_warped_image = registerer.transform_image(moving_image, reg_result.fixed_transform, 'LINEAR')
            aff_warped_mask = registerer.transform_image(moving_mask, reg_result.fixed_transform, 'NN')
            lms = source_section.landmarks.data[['x', 'y']].to_numpy()
            aff_warped_pointset = registerer.transform_pointset(lms, reg_result.moving_transform)
            aff_warped_pointset_df = pd.DataFrame(aff_warped_pointset, columns=['x', 'y'])
            aff_warped_pointset_df['label'] = source_section.landmarks.data.label
            nonrigid_opts = Options()
            nonrigid_opts.affine_do_registration = False
            nonrigid_opts.greedy_opts.s1 = 5
            nonrigid_opts.greedy_opts.s2 = 5
            nonrigid_opts.deformable_do_denoising = False
            start = time.time()
            nr_reg_result = registerer.register(moving_img=aff_warped_image,
                                              fixed_img=fixed_image,
                                              moving_img_mask=aff_warped_mask,
                                              options=nonrigid_opts)
            end = time.time()
            # nr_warped_image = registerer.transform_image(aff_warped_image, nr_reg_result.fixed_transform, 'LINEAR')
            # nr_warped_mask = registerer.transform_image(aff_warped_mask, nr_reg_result.fixed_transform, 'NN')
            aff_lms = aff_warped_pointset_df[['x', 'y']].to_numpy()
            nr_warped_pointset = registerer.transform_pointset(aff_lms, nr_reg_result.moving_transform)
            nr_warped_pointset_df = pd.DataFrame(nr_warped_pointset, columns=['x', 'y'])
            nr_warped_pointset_df['label'] = aff_warped_pointset_df['label']
            target_pointset = target_section.landmarks.data
            lm_subdir = join(lm_dir, f'{source_idx}_{target_idx}')
            target_pointset_path = join(lm_subdir, f'{target_idx}_lms.csv')
            warped_pointset_path = join(lm_subdir, f'{source_idx}_lms_warped.csv')
            create_if_not_exists(lm_subdir)
            nr_warped_pointset_df.to_csv(warped_pointset_path)
            target_pointset.to_csv(target_pointset_path)
            duration = end - start
            shape = target_section.image.data.shape
            mean_rtre, median_rtre, mean_tre, median_tre = compute_tre(target_pointset, nr_warped_pointset_df, shape)            
            mov_x_dim, mov_y_dim = source_section.image.data.shape[:2]
            fix_x_dim, fix_y_dim = target_section.image.data.shape[:2]
            row = {
                'core_name': core_name,
                'fixed_section_id': target_idx,
                'moving_section_id': source_idx,
                'mean_rtre': mean_rtre,
                'median_rtre': median_rtre,
                'mean_tre': mean_tre,
                'median_tre': median_tre,
                'duration': duration,
                'mov_x_dim': mov_x_dim,
                'mov_y_dim': mov_y_dim,
                'fix_x_dim': fix_x_dim,
                'fix_y_dim': fix_y_dim
            }
            row = pd.DataFrame(row, index=[0])
            df = pd.concat([df, row]).reset_index(drop=True)
            print(row)
            print(f'Duration: {duration}.')
        df.to_csv(join(target_dir, 'stats.csv'))
        df_all = pd.concat([df_all, df]).reset_index(drop=True)
    df_all.to_csv(join(root_dir, 'stats.csv'))



            


if __name__ == '__main__':
    # main()
    # main2()
    main4()