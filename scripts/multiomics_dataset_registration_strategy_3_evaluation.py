"""
Stepwise multisection integration. Uses precomputed transformations.
Registration works by "tarzaning" along each section.
"""
import sys
sys.path.append('../../spami')
sys.path.append('../')

import json
import os
from os.path import join, exists
import shutil
import time

import pandas as pd
import SimpleITK as sitk

from greedyfhist.registration import greedy_f_hist

from spami.reg_graph import RegGraph
from spami.utils.utils import create_if_not_exists, filter_node_ids
from spami.utils.image_utils import get_symmetric_padding_for_sections
from spami.utils.metrics import compute_tre
from spami.utils.plot import plot_registration_summary

from greedyfhist.registration.greedy_f_hist import RegResult, GreedyFHist, get_default_args


def load_all_core_transformations(core_name):
    transformations = {}
    section_distances = [1,2,3,4,5]
    for section_distance in section_distances:
        dir_ = f'../save_directories/multisection_integration/direct/multistep_handover_{section_distance}_step_test/{core_name}'
        transformations_ = load_transformations(dir_)
        for key in transformations_:
            transformations[key] = transformations_[key]
    return transformations


def load_transformations(directory):
    """
    Transformations are stored in sub directories in form {target_section}_{source_section}.
    """
    transformations = {}
    for sub_dir in os.listdir(directory):
        sub_dir_ = join(directory, sub_dir, 'out')
        if not os.path.isdir(sub_dir_):
            continue
        transformation = RegResult.from_directory(sub_dir_)
        transformations[sub_dir] = transformation
    return transformations


def load_registerer():
    reg_config = {
        'path_to_greedy': '/mnt/work/workbench/maximilw/applications/test/greedy/build2/'
    } 
    registerer = GreedyFHist.load_from_config(reg_config)
    return registerer


def get_registration_path(transformations, start, end):
    graph_, vertices = get_graph_structure(transformations)
    path = get_path_wrapper(graph_, start, end, vertices)
    reg_path = path_to_transf_seq(path)
    return reg_path


def get_graph_structure(transformations):
    graph = {}
    vertices = set()
    for key in transformations.keys():
        t, s = key.split('_')
        if s not in graph:
            graph[s] = []
        graph[s].append(t)
        vertices.add(t)
        vertices.add(s)
    return graph, vertices


def get_path_wrapper(graph, start, end, vertices):
    if start not in vertices or end not in vertices:
        return []
    return get_path(graph, start, end, [])


def get_path(graph, current_vertex, end_vertex, current_path):
    current_path.append(current_vertex)
    if current_vertex == end_vertex:
        return current_path
    connected_vertices = graph[current_vertex]
    for vertex in connected_vertices:
        if vertex in current_path:
            continue
        path = get_path(graph, vertex, end_vertex, current_path)
        if len(path) == 0:
            continue
        else:
            return path
    return [] 


def path_to_transf_seq(path):
    transf_path = []
    for source, target in zip(path[:-1], path[1:]):
        transf_path.append(f'{target}_{source}')
    return transf_path


def get_core_names():
    return [x.split('.')[0] for x in os.listdir('../../spami/configs/cores_hr/')]


def compute_tre_with_dfs(warped_df, fixed_df, shape):
    unified_lms = compute_distance_for_lm(warped_df, fixed_df)
    diag = np.sqrt(np.square(shape[0]) + np.square(shape[1]))
    unified_lms['rtre'] = unified_lms['tre']/diag
    mean_rtre = np.mean(unified_lms['rtre'])
    median_rtre = np.median(unified_lms['rtre'])
    median_tre = np.median(unified_lms['tre'])
    mean_tre = np.mean(unified_lms['tre'])
    return mean_rtre, median_rtre, mean_tre, median_tre
    

def update_args(args, core_name, fixed_section, moving_section):
    if core_name == '047_02' and fixed_section == 6 and moving_section == 7:
        print('Updating parameters to fit better.')
        args['mov_sr'] = 15
        args['mov_sp'] = 15
        return args
    else:
        print('Not updating args.')
        return args

def get_4_step_paths():
    l = []
    l.append((2, 3, 6))
    l.append((3, 6, 7))
    l.append((6, 7, 10))
    l.append((6, 8, 10))
    l.append((6, 9, 10))
    l.append((7, 8, 11))
    l.append((7, 9, 11))
    l.append((7, 10, 11))
    return l

def get_5_step_paths():
    l = []
    l.append((1, 3, 6))
    l.append((1, 2, 6))
    l.append((2, 3, 7))
    l.append((2, 6, 7))
    l.append((3, 6, 8))
    l.append((3, 7, 8))
    l.append((6, 7, 11))
    l.append((6, 8, 11))
    l.append((6, 9, 11))
    l.append((6, 10, 11))
    return l

def get_3_step_paths():
    l = []
    l.append((6,7,9))
    l.append((6,8,9))
    l.append((7,8,10))
    l.append((7,9,10))
    l.append((8,9,11))
    l.append((8,10,11))
    return l

    
        
def do_registration_stats(root_target_dir, 
                          section_distance=4):
    if section_distance not in [3,4,5]:
        print(f'Section distance {section_distance} is not available for that script.')
        exit()
    paths = []
    if section_distance == 3:
        paths = get_3_step_paths()
    if section_distance == 4:
        paths = get_4_step_paths()
    elif section_distance == 5:
        paths = get_5_step_paths()
    skip_processed_cores = True

    reg_config = {
        'path_to_greedy': '/mnt/work/workbench/maximilw/applications/test/greedy/build2/'
    } 
    registerer = greedy_f_hist.GreedyFHist.load_from_config(reg_config)
    
    if not os.path.exists(root_target_dir):
        os.mkdir(root_target_dir)
    print(f'Current root dir: {root_target_dir}')
    core_names = get_core_names()
    for core_name in core_names:
        transformations = load_all_core_transformations(core_name)
        print(f'Working on core: {core_name}')
        target_dir = os.path.join(root_target_dir, core_name)
        if skip_processed_cores and exists(target_dir):
            print(f'Core: {core_name} was already processed.')
            continue
        create_if_not_exists(target_dir)
        print(target_dir)

        config_path = os.path.join('../../spami/configs/cores_hr/', core_name + '.json')
        print(config_path)

        graph = RegGraph.from_config_path(config_path, remove_additional_data=True)    
       
        df = pd.DataFrame()
        for reg_seq in paths:
            moving_section_id = reg_seq[0]
            fixed_section_id = reg_seq[-1]
        # for moving_section_id in graph.sections:
            start = time.time()
            # fixed_section_id = moving_section_id + section_distance
            if not all([x in graph.sections for x in reg_seq]):
                print(f'Some of the following paths are missing: {reg_seq}.')
            # if moving_section_id not in graph.sections and fixed_section_id not in graph.sections:
                continue
            print(f'Working on moving: {moving_section_id} and {fixed_section_id}.')
            moving_section = graph.sections[moving_section_id].copy()
            fixed_section = graph.sections[fixed_section_id].copy()
            if moving_section.landmarks is None or fixed_section.landmarks is None:
                print('No landmarks found! Skipping..')
                continue
            sub_dir = join(target_dir, '_'.join([f'{x}' for x in reg_seq]))
            create_if_not_exists(sub_dir)
            default_args = greedy_f_hist.get_default_args()
            default_args['output_dir'] = join(sub_dir, 'out')
            default_args['tmp_dir'] = join(sub_dir, 'tmp')
            warped_dir = join(sub_dir, 'warped_section')
            path = path_to_transf_seq(reg_seq)
            print(f'Current path being worked on: {path}')
            warped_section = moving_section.copy()
            warping_successful = True
            for transf_key in path:
                if transf_key not in transformations:
                    print(f'Transformation {transf_key} not found. Aborting this path.')
                    warping_successful = False
                    break
                warped_section = warped_section.warp(registerer, transformations[transf_key], default_args)
            if not warping_successful:
                continue
            # warped_section.store(warped_dir)
            end = time.time()
            print(f'Reg time: {end - start}.')
            mean_rtre, median_rtre, mean_tre, median_tre = compute_tre(fixed_section,
                                                                       warped_section,
                                                                       fixed_section.image.data.shape)
            sitk.WriteImage(sitk.GetImageFromArray(warped_section.image.data), join(sub_dir, 'registered_image.png'))
            row = {
                'core_name': core_name,
                'fixed_section_id': fixed_section_id,
                'moving_section_id': moving_section_id,
                'mean_rtre': mean_rtre,
                'median_rtre': median_rtre,
                'mean_tre': mean_tre,
                'median_tre': median_tre
            }
            print(row)
            row = pd.DataFrame(row, index=[0])
            df = pd.concat([df, row]).reset_index(drop=True)
        df.to_csv(join(target_dir, 'stats.csv'), index=False)


def main():
    section_distances = [3,4,5]
    root_dir = '../save_directories/multisection_integration/direct/multistep_strategy_3/'
    create_if_not_exists(root_dir)
    for section_distance in section_distances:
#        root_transformation_dir = f'../save_directories/multisection_integration/direct/multistep_handover_1_step_test'
        root_target_dir = join(root_dir, f'{section_distance}')
        create_if_not_exists(root_target_dir)
        # root_dir = f'../save_directories/multisection_integration/direct/denoised_10_small_1_step_big_kernel_test3'
        create_if_not_exists(root_dir)
        do_registration_stats(root_target_dir,
                              section_distance=section_distance)
        
if __name__ == '__main__':
    main()
