import logging
import os
from os.path import join
from typing import Any, Dict, List, Optional, Tuple, Union

import geojson
import numpy as np
import pandas as pd
import SimpleITK as sitk
import toml

from greedyfhist.utils.image import read_image
from greedyfhist.utils.io import create_if_not_exists, derive_output_path
from greedyfhist.utils.geojson_utils import read_geojson
from greedyfhist.registration.greedy_f_hist import GreedyFHist, InternalRegParams, RegistrationResult
from greedyfhist.options.options import RegistrationOptions
from greedyfhist.data_types import OMETIFFImage, DefaultImage, Pointset, GeoJsonData, HistologySection


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
    if path.endswith('tiff') or path.endswith('tif'):
        warped_ome_data = OMETIFFImage.load_and_transform_data(path, registerer, transformation)
        return warped_ome_data
    if path.endswith('csv'):
        return Pointset.load_and_transform_data(path, registerer, transformation)
    if path.endswith('geojson'):
        return GeoJsonData.load_and_transform_data(path, registerer, transformation)
    else:
        image_data = DefaultImage.load_and_transform_data(path, registerer, transformation)
        return image_data
    

def load_data(config: Dict) -> Any:
    type_ = config['type']
    if type_ == 'ome'


def resolve_variable(selector: str, 
                    choice1: Optional[Any] = None, 
                    config: Optional[Any] = None,
                    fallback_value: Optional[Any] = None) -> Optional[Any]:
    """Select a variable either from first candidate, from config dictionary or a default value. Return None if neither is found.

    Args:
        selector (str): Key in config.
        choice1 (Optional[Any], optional): First choice to return. Defaults to None.
        config (Optional[Any], optional): Dict containing second choice to return. Defaults to None.
        fallback_value (Optional[Any], optional): Default value if neither choice is present.
        
    Returns:
        Optional[Any]: Returns chosen variable. Returns None if neither variable exists.
    """
    if choice1 is not None:
        return choice1
    if config is None:
        return None
    choice2 = config.get(selector, None)
    if choice2 is None:
        return fallback_value
    return choice2


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
    if path.endswith('tiff') or path.endswith('tif'):
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
                                                   switch_axis=False)
        return image_data


def get_image_type_from_config(config: Dict) -> str:
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
    if type_ in ['tif', 'tiff', 'default']:
        return type_
    else:
        path = config['path']
        if path.endswith('tif') or path.endswith('tiff'):
            return 'tif'
        else:
            return 'default'
        

def guess_and_load_image(path: str, 
                         is_annotation: Optional[bool] = False,
                         switch_axis: Optional[bool] = False) -> Union[OMETIFFImage, DefaultImage]:
    """Guess image type and return guessed image types.

    Args:
        path (str): Path to image file.
        is_annotation: Passed on to loading image.
        switch_axis: Passed on to loading image.

    Returns:
        Union[OMETIFFImage, DefaultImage]: Loaded image.
    """
    if path.endswith('tiff') or path.endswith('tif'):
        return OMETIFFImage.load_from_path(path, is_annotation=is_annotation, switch_axis=switch_axis)
    return DefaultImage.load_from_path(path, is_annotation=is_annotation, switch_axis=switch_axis)




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
    type_ = get_image_type_from_config(config)
    if type_ in ['tiff', 'tif']:
        ome_data = OMETIFFImage.load_from_config(config)
        warped_ome_data = ome_data.transform_data(registerer, transformation)
        return warped_ome_data
    else:
        image_data = DefaultImage.load_from_config(config)
        warped_image_data = image_data.transform_data(registerer,
                                           transformation)
        return warped_image_data


def load_histology_section(image_path: str,
                           additional_images: List[str] = None,
                           additional_annotations: List[str] = None,
                           additional_pointsets: List[str] = None,
                           additional_geojsons: List[str] = None,
                           mask_path: Optional[str] = None):
    image = guess_and_load_image(image_path)
    if mask_path is not None:
        mask = guess_and_load_image(mask_path)
    else:
        mask = None
    additional_data = []
    for path in additional_images:
        ad_image = guess_and_load_image(path)
        additional_data.append(ad_image)
    for path in additional_annotations:
        ad_image = guess_and_load_image(path, is_annotation=True)
        additional_data.append(ad_image)
    for path in additional_pointsets:
        ps = Pointset.load_from_path(path)
        additional_data.append(ps)
    for path in additional_geojsons:
        gs = GeoJsonData.load_from_path(path)
        additional_data.append(gs)
    histology_section = HistologySection(ref_image=image,
                                         ref_mask=None,
                                         additional_data=additional_data)
    return histology_section
        

def register2(moving_image_path: Optional[str] = None,
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
    logging.info('Starting registration process.')
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
    logging.info('Registration options are loaded.')
    if all_paths_are_none([moving_image_path, fixed_image_path, moving_mask_path, fixed_mask_path]):
        moving_image_path, fixed_image_path, moving_mask_path, fixed_mask_path = get_paths_from_config(config.get('input', None))
    if moving_image_path is None and fixed_image_path is None:
        raise Exception('No moving and fixed image path provided!')
    if 'options' in config:
        # TODO: Should I add collision avoidance?
        output_directory = config['options'].get('output_directory', 'out')
        path_to_greedy = config['options'].get('path_to_greedy', '')
    else:
        output_directory = 'out'
        path_to_greedy = ''

    output_directory = resolve_variable('output_directory', output_directory, config.get('options', None), 'out')
    path_to_greedy = resolve_variable('path_to_greedy', path_to_greedy, config.get('options', None), '')
    warp_moving_image = resolve_variable('warp_moving_image', None, config.get('options', None), True)
    save_transform_to_file = resolve_variable('save_transform_to_file', None, config.get('options', None), True)

    moving_histology_section = load_histology_section(
        image_path=moving_image_path,
        additional_images=additional_images,
        additional_annotations=additional_annotations,
        additional_pointsets=additional_pointsets,
        additional_geojsons=additional_geojsons,
        mask_path=moving_mask_path
    )
    
    # Setup file structure
    output_directory_registrations = join(output_directory, 'registrations')
    create_if_not_exists(output_directory_registrations)

    # moving_image = read_image(moving_image_path)
    # moving_mask = read_image(moving_mask_path, True) if moving_mask_path is not None else None

    fixed_image = guess_and_load_image(fixed_image_path)
    fixed_mask = guess_and_load_image(fixed_mask_path, is_annotation=True) if fixed_mask_path is not None else None

    logging.info('Loaded images. Starting registration.')
    logging.info(f'Registration options: {registration_options}')
    registerer = GreedyFHist.load_from_config({'path_to_greedy': path_to_greedy})

    registration_result = registerer.register(
        moving_histology_section.ref_image.data,
        fixed_image.data,
        moving_histology_section.ref_mask.data,
        fixed_mask.data,
        options=registration_options
    )
    logging.info('Registration finished.')
    if save_transform_to_file:
        registration_result.to_file(output_directory_registrations)
        logging.info('Registration saved.')

    

    if warp_moving_image:
        logging.info('Saving warped image.')
        warped_moving_image = moving_image.transform_data(registerer, registration_result)
        output_directory_transformation_data = join(output_directory, 'transformed_data')
        create_if_not_exists(output_directory_transformation_data)
        target_path = derive_output_path(output_directory_transformation_data, os.path.basename(moving_image.path))
        warped_moving_image.to_file(target_path)



    apply_transformation(output_directory=output_directory,
                         images=additional_images,
                         annotations=additional_annotations,
                         pointsets=additional_pointsets,
                         geojsons=additional_geojsons,
                         config=config,
                         registerer=registerer,
                         registration_result=registration_result)
    

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
    logging.info('Starting registration process.')
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
    logging.info('Registration options are loaded.')
    if all_paths_are_none([moving_image_path, fixed_image_path, moving_mask_path, fixed_mask_path]):
        moving_image_path, fixed_image_path, moving_mask_path, fixed_mask_path = get_paths_from_config(config.get('input', None))
    if moving_image_path is None and fixed_image_path is None:
        raise Exception('No moving and fixed image path provided!')
    if 'options' in config:
        # TODO: Should I add collision avoidance?
        output_directory = config['options'].get('output_directory', 'out')
        path_to_greedy = config['options'].get('path_to_greedy', '')
    else:
        output_directory = 'out'
        path_to_greedy = ''

    output_directory = resolve_variable('output_directory', output_directory, config.get('options', None), 'out')
    path_to_greedy = resolve_variable('path_to_greedy', path_to_greedy, config.get('options', None), '')
    warp_moving_image = resolve_variable('warp_moving_image', None, config.get('options', None), True)
    save_transform_to_file = resolve_variable('save_transform_to_file', None, config.get('options', None), True)

    
    # Setup file structure
    output_directory_registrations = join(output_directory, 'registrations')
    create_if_not_exists(output_directory_registrations)

    moving_image = read_image(moving_image_path)
    fixed_image = read_image(fixed_image_path)
    moving_mask = read_image(moving_mask_path, True) if moving_mask_path is not None else None
    fixed_mask = read_image(fixed_mask_path, True) if fixed_mask_path is not None else None

    moving_image = guess_and_load_image(moving_image_path)
    fixed_image = guess_and_load_image(fixed_image_path)
    moving_mask = guess_and_load_image(moving_mask_path) if moving_mask_path is not None else None
    if moving_mask is not None:
        moving_mask = np.squeeze(moving_mask.data)
    fixed_mask = guess_and_load_image(fixed_mask_path) if fixed_mask_path is not None else None
    if fixed_mask is not None:
        fixed_mask = np.squeeze(fixed_mask.data)

    
    logging.info('Loaded images. Starting registration.')
    logging.info(f'Registration options: {registration_options}')
    registerer = GreedyFHist.load_from_config({'path_to_greedy': path_to_greedy})

    registration_result = registerer.register(
        moving_image.data,
        fixed_image.data,
        moving_mask,
        fixed_mask,
        options=registration_options
    )
    logging.info('Registration finished.')
    if save_transform_to_file:
        registration_result.to_file(output_directory_registrations)
        logging.info('Registration saved.')

    if warp_moving_image:
        logging.info('Saving warped image.')
        warped_moving_image = moving_image.transform_data(registerer, registration_result)
        output_directory_transformation_data = join(output_directory, 'transformed_data')
        create_if_not_exists(output_directory_transformation_data)
        target_path = derive_output_path(output_directory_transformation_data, os.path.basename(moving_image.path))
        warped_moving_image.to_file(target_path)



    apply_transformation(output_directory=output_directory,
                         images=additional_images,
                         annotations=additional_annotations,
                         pointsets=additional_pointsets,
                         geojsons=additional_geojsons,
                         config=config,
                         registerer=registerer,
                         registration_result=registration_result)

  

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
    logging.info('Applying transformation to additional data.')
    if registerer is None:
        # We can load GreedyFHist without greedy for this.
        registerer = GreedyFHist.load_from_config({})

    if registration_result is None:
        registration_result = RegistrationResult.load(registration_result_path)

    output_directory_transformed_data = join(output_directory, 'transformed_data')
    create_if_not_exists(output_directory_transformed_data)
    logging.info('Working on commandline arguments.')
    logging.info('Warping images.')
    for path in images:
        warped_data = guess_load_transform_image_data(path, registerer, registration_result, is_annotation=False, switch_axis=False)
        target_path = derive_output_path(output_directory_transformed_data, os.path.basename(path))
        warped_data.to_file(target_path)
    logging.info('Warping annotations.')
    for path in annotations:
        warped_data = guess_load_transform_image_data(path, registerer, registration_result, is_annotation=True, switch_axis=True)
        target_path = derive_output_path(output_directory_transformed_data, os.path.basename(path))
        warped_data.to_file(target_path)
    logging.info('Warping pointsets.')        
    for path in pointsets:
        data = Pointset.load_from_path(path)
        warped_data = Pointset.transform_data(data, registerer, registration_result)
        target_path = derive_output_path(output_directory_transformed_data, os.path.basename(path))
        warped_data.to_file(target_path)   
    logging.info('Warping geojsons.')         
    for path in geojsons:
        data = GeoJsonData.load_from_path(path)
        warped_data = GeoJsonData.transform_data(data, registerer, registration_result)
        target_path = derive_output_path(output_directory_transformed_data, os.path.basename(path))
        warped_data.to_file(target_path)      

    logging.info('Working on arguments from config.')
    logging.info('Warping images and annotations.')
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
    logging.info('Warping pointsets.')    
    additional_pointsets = transform_config.get('pointsets', [])
    for config in additional_pointsets:
        pointset = Pointset.load_data(config)
        warped_pointset = Pointset.transform_data(pointset)
        target_path = derive_output_path(output_directory_transformed_data, os.path.basename(path))
        warped_pointset.to_file(target_path)        
    logging.info('Warping geojsons.')
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