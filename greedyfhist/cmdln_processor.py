import logging
import os
from os.path import join
from typing import Dict, List, Optional, Tuple, Union

import geojson
import numpy as np
import pandas as pd
import SimpleITK as sitk
import toml

from greedyfhist.utils.image import read_image
from greedyfhist.utils.io import create_if_not_exists
from greedyfhist.utils.geojson_utils import read_geojson
from greedyfhist.registration.greedy_f_hist import GreedyFHist, InternalRegParams, RegistrationResult
from greedyfhist.options.options import RegistrationOptions
from greedyfhist.data_types import OMETIFFImage, DefaultImage, Pointset, GeoJsonData


logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)


def all_paths_are_none(path_list: List[str]) -> bool:
    """Checks whether all paths are None.

    Args:
        path_list (List[str]): List of file paths.

    Returns:
        bool: 
    """
    return all(map(lambda x: x is None, path_list))


def get_paths_from_config(config: Optional[Dict] = None) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Extracts moving and fixed image file paths from config.

    Args:
        config (Optional[Dict], optional): Registration config. Defaults to None.

    Returns:
        Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]: Found filepaths.
    """
    if config is None:
        return None, None, None, None
    moving_image_path = config.get('moving_image', None)
    fixed_image_path = config.get('fixed_image', None)
    moving_mask_path = config.get('moving_mask', None)
    fixed_mask_path = config.get('fixed_mask', None)
    return moving_image_path, fixed_image_path, moving_mask_path, fixed_mask_path
        

def guess_load_transform_data(path: str,
                              registerer: GreedyFHist,
                              transformation: RegistrationResult) -> Union[OMETIFFImage, DefaultImage, Pointset, GeoJsonData]:
    """Utility function that determines filetype from path, then applies
    the transformation and returns the warped data.

    Args:
        path (str): Filename
        registerer (GreedyFHist): Initialized GreedyFHist registerer.
        transformation (RegistrationResult): Computed transformation from moving to fixed image.

    Returns:
        _type_: Warped data.
    """
    if path.endswith('ome.tiff') or path.endswith('ome.tif'):
        warped_ome_data = OMETIFFImage.load_and_transform_data(path, registerer, transformation)
        return warped_ome_data
    if path.endswith('csv'):
        return Pointset.load_and_transform_data(path, registerer, transformation)
    if path.endswith('geojson'):
        return GeoJsonData.load_and_transform_data(path, registerer, transformation)
    else:
        image_data = DefaultImage.load_and_transform_data(path, registerer, transformation)
        return image_data


def guess_load_transform_image_data(path: str, 
                              registerer: GreedyFHist,
                              transformation: RegistrationResult,
                              is_annotation: bool = False, 
                              switch_axis: bool = False) -> Union[OMETIFFImage, DefaultImage]:
    """Utility function that loads images based on file ending. Image
    data can be interpreted as either image or annotation and axis can
    be switched (might be necessary for multilabel outputs from QuPath).
    Then applies transformation on image data. 

    Args:
        path (str): Filepath
        registerer (GreedyFHist): Initialized GreedyFHist registerer.
        transformation (RegistrationResult): Computed transformation from moving to fixed image space.
        is_annotation (bool, optional): If True, sets image data to annotation. Defaults to False.
        switch_axis (bool, optional): If True, switches axis 0 and 2. Defaults to False.

    Returns:
        Union[OMETIFFImage, DefaultImage]: Warped imge data.
    """
    if path.endswith('ome.tiff') or path.endswith('ome.tif'):
        ome_data = OMETIFFImage.load_and_transform_data(path, 
                                                        registerer,
                                                        transformation,
                                                        is_annotation=is_annotation, 
                                                        switch_axis=switch_axis)
        return ome_data
    else:
        image_data = DefaultImage.load_and_transform_data(path, 
                                                   registerer,
                                                   transformation,
                                                   is_annotation=is_annotation, 
                                                   switch_axis=switch_axis)
        return image_data


def get_type_from_config(config: Dict) -> str:
    """Derive image type from given config.
    First, tries to use the type key in config.
    If that isnt found, uses the filenames to 
    guess the correct filepath.

    Args:
        config (Dict): 

    Returns:
        str: 
    """
    type_ = config.get('type', None)
    if type_ in ['ome.tif', 'ome.tiff', 'default']:
        return type_
    else:
        path = config['path']
        if path.endswith('ome.tif') or path.endswith('ome.tiff'):
            return 'ome.tif'
        else:
            return 'default'


def guess_load_transform_image_from_config(config: Dict, 
                              registerer: GreedyFHist,
                              transformation: RegistrationResult) -> Union[OMETIFFImage, DefaultImage]:
    """Utility function that loads images based on config object. Image
    data can be interpreted as either image or annotation and axis can
    be switched (might be necessary for multilabel outputs from QuPath).
    Then applies transformation on image data. 

    Args:
        config (Dict): Config of containing image relevant information.
        registerer (GreedyFHist): Initialized GreedyFHist instance.
        transformation (RegistrationResult): Transformation for warping from moving to fixed image space.

    Returns:
        Union[OMETIFFImage, DefaultImage]: Warped data.
    """
    type_ = get_type_from_config(config)
    if type_ in ['ome.tiff', 'ome.tif']:
        ome_data = OMETIFFImage.load_from_config(config)
        warped_ome_data = OMETIFFImage.transform_data(ome_data, registerer, transformation)
        return warped_ome_data
    else:
        image_data = DefaultImage.load_data_from_config(config)
        warped_image_data = DefaultImage.transform_data(image_data,
                                           registerer,
                                           transformation)
        return warped_image_data


def derive_output_path(directory: str, fname: str, limit: int = 1000) -> str:
    """Generates a unique output path. If path is already existing,
    adds a counter value until a unique path is found.

    Args:
        directory (str): target directory
        fname (str): target filename
        limit (int, optional): Limit number to prevent endless loops. Defaults to 1000.

    Returns:
        str: Target path
    """
    target_path = join(directory, fname)
    if not os.path.exists(target_path):
        return target_path
    for suffix in range(limit):
        new_target_path = f'{target_path}_{suffix}'
        if not os.path.exists(new_target_path):
            return new_target_path
    return target_path

def register(moving_image_path: Optional[str] = None,
             fixed_image_path: Optional[str] = None,
             output_directory: Optional[str] = None,
             moving_mask_path: Optional[str] = None,
             fixed_mask_path: Optional[str] = None,
             path_to_greedy: Optional[str] = None,
             config_path: Optional[str] = None,
             additional_images: Optional[List[str]] = None,
             additional_annotations: Optional[List[str]] = None,
             additional_pointsets: Optional[List[str]] = None,
             additional_geojsons: Optional[List[str]] = None):
    """Performs GreedyFHist registration between moving and fixed image followed by
    transformation of provided data. GreedyFHist parameters are read from the 
    config file. Otherwise, uses default. Optionally, masks are loaded included
    in the registration. After registration, additionally provided data is transformed.


    Args:
        moving_image_path (Optional[str], optional): Defaults to None.
        fixed_image_path (Optional[str], optional): Defaults to None.
        output_directory (Optional[str], optional): Defaults to None.
        moving_mask_path (Optional[str], optional): Defaults to None.
        fixed_mask_path (Optional[str], optional): Defaults to None.
        path_to_greedy (Optional[str], optional): Defaults to None.
        config_path (Optional[str], optional): Defaults to None.
        additional_images (Optional[List[str]], optional): Defaults to None.
        additional_annotations (Optional[List[str]], optional): Defaults to None.
        additional_pointsets (Optional[List[str]], optional): Defaults to None.
        additional_geojsons (Optional[List[str]], optional): Defaults to None.
    """
    logging.info('Starting registration.')
    if additional_images is None:
        additional_images = []
    if additional_annotations is None:
        additional_annotations = []
    if additional_pointsets is None:
        additional_pointsets = []
    if additional_geojsons is None:
        additional_geojsons = []
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

    apply_transformation(registration_result,
                         output_directory,
                         additional_images,
                         additional_annotations,
                         additional_pointsets,
                         additional_geojsons,
                         config,
                         registerer)

  

def apply_transformation(output_directory: str,
                         images: List[str],
                         annotations: List[str],
                         pointsets: List[str],
                         geojsons: List[str],
                         config: Dict,
                         registerer: GreedyFHist = None,
                         registration_result: RegistrationResult = None,
                         registration_result_path: str = None):
    """Applies registration_result to provided data. If registration_result
    is None, a registration_result is read from file. Registration_result
    is first applied to data provided as command line arguments (images,
    annotations, pointsets, geojsons). Then data declared in the config 
    file is transformed. All transformed data is stored in 
    'output_directory'/transformed_data.

    Args:
        output_directory (str): _description_
        images (List[str]): _description_
        annotations (List[str]): _description_
        pointsets (List[str]): _description_
        geojsons (List[str]): _description_
        config (List[str]): _description_
        registerer (List[str], optional): _description_. Defaults to None.
        registration_result (RegistrationResult, optional): _description_. Defaults to None.
        registration_result_path (str, optional): _description_. Defaults to None.
    """
    
    if registerer is None:
        # We can load GreedyFHist without greedy for this.
        registerer = GreedyFHist.load_from_config({})

    if registration_result is None:
        registration_result = RegistrationResult.load(registration_result_path)

    output_directory_transformed_data = join(output_directory, 'transformed_data')
    create_if_not_exists(output_directory_transformed_data)
    for path in images:
        warped_data = guess_load_transform_image_data(path, registerer, registration_result, is_annotation=False, switch_axis=False)
        target_path = derive_output_path(output_directory_transformed_data, os.path.basename(path))
        warped_data.to_file(target_path)
    for path in annotations:
        warped_data = guess_load_transform_image_data(path, is_annotation=True, switch_axis=True)
        target_path = derive_output_path(output_directory_transformed_data, os.path.basename(path))
        warped_data.to_file(target_path)
    for path in pointsets:
        data = Pointset.load_from_path(path)
        warped_data = Pointset.transform_data(data)
        target_path = derive_output_path(output_directory_transformed_data, os.path.basename(path))
        warped_data.to_file(target_path)        
    for path in geojsons:
        data = GeoJsonData.load_from_path(path)
        warped_data = GeoJsonData.transform_data(data)
        target_path = derive_output_path(output_directory_transformed_data, os.path.basename(path))
        warped_data.to_file(target_path)      

    if 'input' not in config:
        return    
    transform_config = config['input'].get('transform', None)
    if transform_config is None:
        return     
    additional_images = transform_config.get('images', [])
    for config in additional_images:
        warped_data = guess_load_transform_image_from_config(config, registerer, registration_result)
        target_path = derive_output_path(output_directory_transformed_data, os.path.basename(path))
        warped_data.to_file(target_path)
    additional_pointsets = transform_config.get('pointsets', [])
    for config in additional_pointsets:
        pointset = Pointset.load_data(config)
        warped_pointset = Pointset.transform_data(pointset)
        target_path = derive_output_path(output_directory_transformed_data, os.path.basename(path))
        warped_pointset.to_file(target_path)        
    additional_geojsons = transform_config.get('geojsons', [])
    for path in additional_geojsons:
        data = GeoJsonData.load_from_path(path)
        warped_data = GeoJsonData.transform_data(data)
        target_path = derive_output_path(output_directory_transformed_data, os.path.basename(path))
        warped_data.to_file(target_path)               


def groupwise_registration(source_directory: str = None,
                           output_directory: str = None,
                           config_path: str = None):
    pass