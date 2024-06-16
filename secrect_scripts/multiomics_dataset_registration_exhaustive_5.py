"""
Stepwise multisection integration. Uses precomputed transformations.
Registration works by "tarzaning" along each section.
"""
import sys
sys.path.append('../../spami')
sys.path.append('../')
from itertools import permutations
import json
import os
from os.path import join, exists
import shutil
import time

import numpy as np
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
    return [x.split('.')[0] for x in os.listdir('../../spami/configs/cores_hr/') if x.endswith('.json')]


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

def get_multiomics_integration_path():
    l = []
    l.append((2,7))
    l.append((2,3,7))
    l.append((2,6,7))
    l.append((2,3,6,7))
    return l

def get_already_processed_paths():
    l = []
    l.append((1, 6))
    l.append((2, 7))
    l.append((3, 8))
    l.append((6, 11))
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
    l.append((6,7,8,9,10,11))
    return l


def get_all_paths(s, e, nodes, filter_out_already_processed_paths=True):
    l = list(range(s, e+1))
    l = [x for x in l if x in nodes]
    all_paths = []
    for i in range(2, len(l)+1):
        sub_paths = list(permutations(l, i))
        sub_paths = [x for x in sub_paths if x[0] == s and x[-1] == e]
        sub_paths = [x for x in sub_paths if is_ordered(x)]
        all_paths += sub_paths
    if filter_out_already_processed_paths:
        already_processed_paths = get_already_processed_paths()
        all_paths = [x for x in all_paths if x not in already_processed_paths]
    return all_paths


def is_ordered(l):
    for i in range(len(l)-1):
        if l[i] > l[i+1]:
            return False
    return True

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

def load_all_core_transformations(core_name):
    transformations = {}
    section_distances = [1,2,3,4,5]
    for section_distance in section_distances:
        dir_ = f'../save_directories/multisection_integration/direct/multistep_handover_{section_distance}_step_test/{core_name}'
        transformations_ = load_transformations(dir_)
        for key in transformations_:
            transformations[key] = transformations_[key]
    return transformations


        
def do_registration_stats(root_target_dir):
    skip_processed_cores = False
    section_distance = 5
    reg_config = {
        'path_to_greedy': '/mnt/work/workbench/maximilw/applications/test/greedy/build2/'
    } 
    registerer = greedy_f_hist.GreedyFHist.load_from_config(reg_config)
    already_processed_paths = get_already_processed_paths()
    if not os.path.exists(root_target_dir):
        os.mkdir(root_target_dir)
    print(f'Current root dir: {root_target_dir}')
    core_names = get_core_names()
    for core_name in core_names:
        if core_name in ['003_01']:
            continue
        print(f'Working on core: {core_name}')
        target_dir = os.path.join(root_target_dir, core_name)
        if skip_processed_cores and exists(target_dir):
            print(f'Core: {core_name} was already processed.')
            continue
        create_if_not_exists(target_dir)
        print(f'Target dir: {target_dir}')

        config_path = os.path.join('../../spami/configs/cores_hr/', core_name + '.json')
        print(config_path)

        graph = RegGraph.from_config_path(config_path, remove_additional_data=True)    
        nodes = [x for x in graph.sections if x > 3]
        core_df = pd.DataFrame()
        transformations = load_all_core_transformations(core_name)
        for section_id in nodes:
            start_section = section_id
            end_section = section_id + section_distance
            section_pair_dir = join(target_dir, f'{start_section}_{end_section}')
            if end_section not in graph.sections:
                continue
            all_paths = get_all_paths(start_section, end_section, nodes, filter_out_already_processed_paths=False)
            create_if_not_exists(section_pair_dir)                
            df = pd.DataFrame()
            print(all_paths)
            for reg_seq in all_paths:
                n_sections = len(reg_seq)
                moving_section_id = reg_seq[0]
                fixed_section_id = reg_seq[-1]
                start = time.time()
                print(f'Working on: {reg_seq}.')
                # print(f'Working on moving: {moving_section_id} and {fixed_section_id}.')            
                if moving_section_id not in graph.sections and fixed_section_id not in graph.sections:
                    print(f'Source and target not found for seg: {reg_seq}.')
                    continue
                # reg_seq = [x for x in reg_seq if x in graph.sections]
                moving_section = graph.sections[moving_section_id].copy()
                fixed_section = graph.sections[fixed_section_id].copy()
                if moving_section.landmarks is None or fixed_section.landmarks is None:
                    print('No landmarks found! Skipping..')
                    continue
                sub_dir = join(section_pair_dir, '_'.join([f'{x}' for x in reg_seq]))
                create_if_not_exists(sub_dir)
                default_args = greedy_f_hist.get_default_args()
                default_args['output_dir'] = join(sub_dir, 'out')
                default_args['tmp_dir'] = join(sub_dir, 'tmp')
                warped_dir = join(sub_dir, 'warped_section')
                warped_section = graph.sections[reg_seq[0]].copy()
                start = time.time()
                # path = get_registration_path(transformations, str(moving_section_id), str(fixed_section_id))
                path = path_to_transf_seq(reg_seq)
                print(f'Path of transformations: {path}.')
                if len(path) == 0:
                    print('No path found for current pair.')
                    continue
                warped_section = moving_section.copy()
                for transf_key in path:
                    print(f'Now warping key: {transf_key}.')
                    warped_section = warped_section.warp(registerer, transformations[transf_key], default_args)                
                # for node in reg_seq[1:]:
                #     print(f'Now on node: {node}.')
                #     fixed_section = graph.sections[node].copy()
                #     transformation = registerer.register(warped_section.image.data,
                #                                          fixed_section.image.data,
                #                                          warped_section.segmentation_mask.data,
                #                                          fixed_section.segmentation_mask.data,
                #                                          default_args)
                #     warped_section = warped_section.warp(registerer, transformation, default_args) # Warp section here 
                mean_rtre, median_rtre, mean_tre, median_tre = compute_tre(fixed_section,
                                                                           warped_section,
                                                                           fixed_section.image.data.shape)    
                end = time.time()
                print(f'Reg time: {end - start}.')                

                row = {
                    'core_name': core_name,
                    'fixed_section_id': fixed_section_id,
                    'moving_section_id': moving_section_id,
                    'mean_rtre': mean_rtre,
                    'median_rtre': median_rtre,
                    'mean_tre': mean_tre,
                    'median_tre': median_tre,
                    'path': ','.join([str(x) for x in reg_seq]),
                    'n_sections': n_sections
                }
                print(row)
                row = pd.DataFrame(row, index=[0])
                df = pd.concat([df, row]).reset_index(drop=True)
            df.to_csv(join(section_pair_dir, 'stats.csv'),index=False)
            core_df = pd.concat([core_df, df]).reset_index(drop=True)
        core_df.to_csv(join(target_dir, 'stats.csv'), index=False)


def main():
    # section_distances = [3,4,5]
    #root_dir = '../save_directories/multisection_integration/direct/multiomics_registration_exhaustive/'
    root_dir = '/mnt/scratch/maximilw/multiomics_integration/multiomics_registration_exhaustive2'
    create_if_not_exists(root_dir)
    do_registration_stats(root_dir)
        
if __name__ == '__main__':
    main()
