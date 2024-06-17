import logging
import os
from os.path import join
from typing import Dict, List, Optional, Tuple

import geojson
import numpy as np
import pandas as pd
import SimpleITK as sitk
import toml

from greedyfhist.utils.image import read_image
from greedyfhist.utils.io import create_if_not_exists, read_config
from greedyfhist.utils.geojson_utils import read_geojson
from greedyfhist.registration.greedy_f_hist import GreedyFHist, RegResult
from greedyfhist.options.options import RegistrationOptions


logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)

def all_paths_are_none(path_list: List[str]) -> bool:
    return all(map(lambda x: x is None, path_list))

def get_paths_from_config(config: Optional[Dict] = None) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    if config is None:
        return None, None, None, None
    moving_image_path = config.get('moving_image', None)
    fixed_image_path = config.get('fixed_image', None)
    moving_mask_path = config.get('moving_mask', None)
    fixed_mask_path = config.get('fixed_mask', None)
    return moving_image_path, fixed_image_path, moving_mask_path, fixed_mask_path

def register(moving_image_path=None,
             fixed_image_path=None,
             output_directory=None,
             moving_mask_path=None,
             fixed_mask_path=None,
             path_to_greedy=None,
             config_path=None):
    """
    Works as follows:
    1. Loads everything from config
    2. Override everything from config with manually specified args.
    """
    logging.info('Starting registration.')
    if config_path is not None:
        with open(config_path) as f:
            config = toml.load(f)
    else:
        config = {}
    if 'gfh_options' in config:
        registration_options = RegistrationOptions.parse_cmdln_dict(config['gfh_options'])
    else:
        registration_options = RegistrationOptions.default_options()
    if all_paths_are_none([moving_image_path, fixed_image_path, moving_mask_path, fixed_mask_path]):
        moving_image_path, fixed_image_path, moving_mask_path, fixed_mask_path = get_paths_from_config(config.get('input', None))
    if moving_image_path is None and fixed_image_path is None:
        pass # Throw error
    if 'options' in config:
        # TODO: Should I add collision avoidance?
        output_directory = config['options'].get('output_directory', 'out')
        path_to_greedy = config['options'].get('path_to_greedy', '')
    else:
        output_directory = 'out'
        path_to_greedy = ''
    
    # Setup file structure
    output_directory_registrations = join(output_directory, 'registrations')
    create_if_not_exists(output_directory_registrations)
        
    
    moving_image = read_image(moving_image_path)
    fixed_image = read_image(fixed_image_path)
    moving_mask = read_image(moving_mask_path, True) if moving_mask_path is not None else None
    fixed_mask = read_image(fixed_mask_path, True) if fixed_mask_path is not None else None
    
    registerer = GreedyFHist.load_from_config({'path_to_greedy': path_to_greedy})

    registration_result = registerer.register(
        moving_image,
        fixed_image,
        moving_mask,
        fixed_mask,
        options=registration_options
    )
    registration_result.to_file(output_directory_registrations)


def transform(transformation,
         output_directory,
         path_to_greedy,
         images,
         annotations,
         coordinates,
         geojsons):
    transformation_ = RegResult.from_directory(transformation)
    config = {
        'path_to_greedy': path_to_greedy
    }
    greedy_f_hist_ = GreedyFHist.load_from_config(config)
    create_if_not_exists(output_directory)
    tmp_directory = join(output_directory, 'tmp')
    create_if_not_exists(tmp_directory)
    # TODO: Add mechanism to resolve name conflicts or warn at least.
    for image_path in images:
        image_ = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
        warped_image = greedy_f_hist_.warp_image(image_, transformation_, 'LINEAR', {'tmp_dir': tmp_directory})
        sitk_warped_image = sitk.GetImageFromArray(warped_image.registered_image)
        # Write to file.
        image_name = os.path.basename(image_path).rsplit('.', maxsplit=1)[0]
        out_path = join(output_directory, f'{image_name}.nii.gz')
        sitk.WriteImage(sitk_warped_image, out_path)
    for annotation_path in annotations:
        # print('ann before1')
        annotations_ = sitk.GetArrayFromImage(sitk.ReadImage(annotation_path))
        # Annotations are assumed to be in shape (N annotations)xWIDTHxHEIGHT
        # Solve this issue by switching axes.
        annotations_ = np.moveaxis(annotations_, 0, -1)
        # print(f'Annotation shape before warping: {annotations_.shape}')
        warped_annotations_result = greedy_f_hist_.warp_image(annotations_, transformation_, 'NN', {'tmp_dir': tmp_directory})
        warped_annotations = warped_annotations_result.registered_image
        # Switch axis back to ensure integrity.
        # print(f'Annotation shape after warping: {warped_annotations.shape}')
        warped_annotations = np.moveaxis(warped_annotations, -1, 0)
        # print(f'Annotation shape after reshaping: {warped_annotations.shape}')

        # Write to file.
        sitk_warped_annotations = sitk.GetImageFromArray(warped_annotations)
        annotations_name = os.path.basename(annotation_path).rsplit('.', maxsplit=1)[0]
        out_path = join(output_directory, f'{annotations_name}.nii.gz')
        sitk.WriteImage(sitk_warped_annotations, out_path)
        # print('ann before2')
    for coordinates_path in coordinates:
        coordinates_ = pd.read_csv(coordinates_path, index_col=0)
        warped_coordinates_res = greedy_f_hist_.warp_coordinates(coordinates_, transformation_, {'tmp_dir': tmp_directory})
        warped_coordinates = warped_coordinates_res.pointcloud
        coordinates_name = os.path.basename(coordinates_path).rsplit('.', maxsplit=1)[0]
        out_path = join(output_directory, f'{coordinates_name}.csv')
        warped_coordinates.to_csv(out_path)
    for geojson_path in geojsons:
        geojson_data = read_geojson(geojson_path)
        warped_geojson_data, _ = greedy_f_hist_.transform_geojson(geojson_data, transformation_, {'tmp_dir': tmp_directory})
        geojson_name = os.path.basename(geojson_path).rsplit('.', maxsplit=1)[0]
        out_path = join(output_directory, f'{geojson_name}.geojson')
        with open(out_path, 'w') as f:
            geojson.dump(warped_geojson_data, f)


def register_by_config(config_path):
    config = read_config(config_path)
    path_to_greedy = config.get('path_to_greedy', None)
    if config['mode'] == 'multi-step-registration':
        # Setup directory structure
        output_directory = config_path['output_directory']
        create_if_not_exists(output_directory)
        output_reg_dir = join(output_directory, 'registration')
        create_if_not_exists(output_reg_dir)
        output_transform_dir = join(output_directory, 'transforms')
        create_if_not_exists(output_transform_dir)
        image_configs = config['data']
        n_regs = len(image_configs) - 1
        # Load data
        moving_image = config['data'][0]['image']
        moving_mask = config['data'][0].get('mask', None)
        moving_image_name = os.path.basename(moving_image).rsplit('.', maxsplit=1)[0]
        moving_mask_name = os.path.basename(moving_mask).rsplit('.', maxsplit=1)[0] if moving_mask is not None else None
        for idx, image_config in enumerate(image_configs[1:]):
            create_if_not_exists(output_directory)
            cur_output_reg_dir = f'{output_reg_dir}/{idx}'
            fixed_image = image_config['image']
            fixed_mask = image_config.get('mask', None)
            register(moving_image,
                fixed_image,
                cur_output_reg_dir,
                moving_mask=moving_mask,
                fixed_mask=fixed_mask,
                path_to_greedy=path_to_greedy,
                is_cmd_line=False)
            transform_path = cur_output_reg_dir
            cur_output_trf_dir = f'{output_transform_dir}/{idx}'
            transform(transform_path,
                cur_output_trf_dir,
                path_to_greedy,
                moving_image,
                annotations=[moving_mask])
            moving_image = join(cur_output_trf_dir, moving_image_name)
            moving_mask = join(cur_output_trf_dir, moving_mask_name) if moving_mask is not None else None
        # TODO: Continue here!!!
        # TODO: Add support for transforming multiregistrations.