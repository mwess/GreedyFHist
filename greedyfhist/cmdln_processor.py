import logging
from os.path import join
from typing import Any, Dict, List, Optional, Tuple, Union

import toml

from greedyfhist.utils.io import create_if_not_exists
from greedyfhist.registration.greedy_f_hist import GreedyFHist, RegistrationResult
from greedyfhist.options.options import RegistrationOptions
from greedyfhist.data_types import OMETIFFImage, DefaultImage, Pointset, GeoJsonData, HistologySection


logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.WARN)


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
                                                   keep_axis=False)
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
                         keep_axis: Optional[bool] = False) -> Union[OMETIFFImage, DefaultImage]:
    """Guess image type and return guessed image types.

    Args:
        path (str): Path to image file.
        is_annotation: Passed on to loading image.
        keep_axis: Passed on to loading image.

    Returns:
        Union[OMETIFFImage, DefaultImage]: Loaded image.
    """
    if path.endswith('tiff') or path.endswith('tif'):
        return OMETIFFImage.load_from_path(path, is_annotation=is_annotation, keep_axis=False)
    return DefaultImage.load_from_path(path, is_annotation=is_annotation, keep_axis=True)


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
    image = load_image_from_config(config)
    warped_image = image.transform_data(registerer, transformation)
    return warped_image

    
def load_image_from_config(config: Dict) -> Union[DefaultImage, OMETIFFImage]:
    """Load image based on config.

    Args:
        config (Dict):

    Returns:
        Union[DefaultImage, OMETIFFIMage]: 
    """
    type_ = get_image_type_from_config(config)
    if type_ in ['tiff', 'tif']:
        return OMETIFFImage.load_data(config)
    return DefaultImage.load_data(config)


def guess_type(path: str) -> str:
    if path.endswith('geojson'):
        return 'geojson'
    if path.endswith('csv'):
        return 'pointset'
    if path.endswith('tiff') or path.endswith('tif'):
        return 'tiff'
    return 'default'


def load_data_from_config(config: Dict) -> Union[DefaultImage, OMETIFFImage, Pointset, GeoJsonData]:
    type_ = config.get('type', None)
    if type_ is None:
        type_ = guess_type(config['path'])
    if type_ == 'geojson':
        return GeoJsonData.load_data(config)
    if type_ == 'pointset':
        return Pointset.load_data(config)
    if type_ in ['tif', 'tiff']:
        return OMETIFFImage.load_data(config)
    if type_ == 'default':
        return DefaultImage.load_data(config)
    # Throw an error message otherwise.


def load_sections(section_configs: Dict) -> List[HistologySection]:
    sections = []
    sorted_keys = sorted(section_configs.keys(), key=lambda x: int(x.replace('section', '')))
    for key in sorted_keys:
        section_config = section_configs[key]
        histology_section = load_histology_section_from_config(section_config)
        sections.append(histology_section)
    return sections


def load_histology_section_from_config(config: Dict) -> HistologySection:
    ref_img_config = config.get('reference_image', None)
    if ref_img_config is not None:
        reference_image = load_image_from_config(ref_img_config)
    else:
        reference_image = None
    ref_mask_config = config.get('reference_mask', None)
    if ref_mask_config is not None:
        config['is_annotation'] = True
        reference_mask = load_image_from_config(ref_mask_config)
    else:
        reference_mask = None
    additional_data_config = config.get('additional_data', [])
    ad_data = []
    for ad_config in additional_data_config:
        data = load_data_from_config(ad_config)
        ad_data.append(data)
    return HistologySection(
        ref_image=reference_image,
        ref_mask=reference_mask,
        additional_data=ad_data
    )


def load_base_histology_section(image_path: Optional[str] = None,
                                mask_path: Optional[str] = None):
    if image_path is not None:
        image = guess_and_load_image(image_path)
    else:
        image = None
    if mask_path is not None:
        mask = guess_and_load_image(mask_path)
    else:
        mask = None
    histology_section = HistologySection(ref_image=image,
                                         ref_mask=mask)
    return histology_section


def load_histology_section(image_path: Optional[str] = None,
                           additional_images: List[str] = None,
                           additional_annotations: List[str] = None,
                           additional_pointsets: List[str] = None,
                           additional_geojsons: List[str] = None,
                           mask_path: Optional[str] = None):
    if image_path is not None:
        image = guess_and_load_image(image_path)
    else:
        image = None
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
                                         ref_mask=mask,
                                         additional_data=additional_data)
    return histology_section
        

def register(moving_image_path: Optional[str] = None,
             fixed_image_path: Optional[str] = None,
             output_directory: Optional[str] = None,
             moving_mask_path: Optional[str] = None,
             fixed_mask_path: Optional[str] = None,
             path_to_greedy: Optional[str] = None,
             config_path: Optional[str] = None,
             default_images: Optional[List[str]] = None,
             default_annotations: Optional[List[str]] = None,
             tif_images: Optional[List[str]] = None,
             tif_annotations: Optional[List[str]] = None,
             pointsets: Optional[List[str]] = None,
             geojsons: Optional[List[str]] = None):
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
    if default_images is None:
        default_images = []
    if default_annotations is None:
        default_annotations = []
    if tif_images is None:
        tif_images = []
    if tif_annotations is None:
        tif_annotations = []
    if pointsets is None:
        pointsets = []
    if geojsons is None:
        geojsons = []

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

    output_directory = resolve_variable('output_directory', output_directory, config.get('options', None), 'out')
    path_to_greedy = resolve_variable('path_to_greedy', path_to_greedy, config.get('options', None), '')
    save_transform_to_file = resolve_variable('save_transform_to_file', None, config.get('options', None), True)

    moving_histology_section = load_base_histology_section(
        image_path=moving_image_path,
        mask_path=moving_mask_path
    )
    
    for default_image_path in default_images:
        default_image = DefaultImage.load_from_path(default_image_path)
        moving_histology_section.additional_data.append(default_image)
    for default_annotation_path in default_annotations:
        default_annotation = DefaultImage.load_from_path(default_annotation_path, is_annotation=True)
        moving_histology_section.additional_data.append(default_annotation)
    for tif_image_path in tif_images:
        tif_image = OMETIFFImage.load_from_path(tif_image_path)
        moving_histology_section.additional_data.append(tif_image)
    for tif_annotation_path in tif_annotations:
        tif_annotation = OMETIFFImage.load_from_path(tif_annotation_path, is_annotation=True)
        moving_histology_section.additional_data.append(tif_annotation)
    for pointset_path in pointsets:
        pointset = Pointset.load_from_path(pointset_path)
        moving_histology_section.additional_data.append(pointset)
    for geojson_path in geojsons:
        geojson_data = GeoJsonData.load_from_path(geojson_path)
        moving_histology_section.additional_data.append(geojson_data)


    # Setup file structure
    output_directory_registrations = join(output_directory, 'registrations')
    create_if_not_exists(output_directory_registrations)

    fixed_image = guess_and_load_image(fixed_image_path)
    fixed_mask = guess_and_load_image(fixed_mask_path, is_annotation=True) if fixed_mask_path is not None else None

    logging.info('Loaded images. Starting registration.')
    logging.info(f'Registration options: {registration_options}')
    registerer = GreedyFHist.load_from_config({'path_to_greedy': path_to_greedy})

    moving_mask = moving_histology_section.ref_mask.data if moving_histology_section.ref_mask is not None else None
    fixed_mask = fixed_mask.data if fixed_mask is not None else None
    registration_result = registerer.register(
        moving_histology_section.ref_image.data,
        fixed_image.data,
        moving_mask,
        fixed_mask,
        options=registration_options
    )
    logging.info('Registration finished.')
    if save_transform_to_file:
        registration_result.to_file(output_directory_registrations)
        logging.info('Registration saved.')

    
    output_directory_transformation_data = join(output_directory, 'transformed_data')
    create_if_not_exists(output_directory_transformation_data)

    warped_histology_section = moving_histology_section.apply_transformation(registration_result=registration_result,
                                                                             registerer=registerer)
    warped_histology_section.to_directory(output_directory_transformation_data)

  
def apply_transformation(
             output_directory: Optional[str] = None,
             config_path: Optional[str] = None,
             path_to_transform: Optional[str] = None,
             default_images: Optional[List[str]] = None,
             default_annotations: Optional[List[str]] = None,
             tif_images: Optional[List[str]] = None,
             tif_annotations: Optional[List[str]] = None,
             pointsets: Optional[List[str]] = None,
             geojsons: Optional[List[str]] = None):
    """Transforms provided data from moving to fixed image space by
    load a computed transformation. If registration_result
    is None, a registration_result is read from file. Registration_result
    is first applied to data provided as command line arguments (images,
    annotations, pointsets, geojsons). Then data declared in the config 
    file is transformed. All transformed data is stored in 
    'output_directory'/transformed_data.


    Args:
        moving_image_path (Optional[str], optional): Defaults to None.
        fixed_image_path (Optional[str], optional): Defaults to None.
        output_directory (Optional[str], optional): Defaults to None.
        moving_mask_path (Optional[str], optional): Defaults to None.
        fixed_mask_path (Optional[str], optional): Defaults to None.
        config_path (Optional[str], optional): Defaults to None.
        additional_images (Optional[List[str]], optional): Defaults to None.
        additional_annotations (Optional[List[str]], optional): Defaults to None.
        additional_pointsets (Optional[List[str]], optional): Defaults to None.
        additional_geojsons (Optional[List[str]], optional): Defaults to None.
    """
    if default_images is None:
        default_images = []
    if default_annotations is None:
        default_annotations = []
    if tif_images is None:
        tif_images = []
    if tif_annotations is None:
        tif_annotations = []
    if pointsets is None:
        pointsets = []
    if geojsons is None:
        geojsons = []
    
    if config_path is not None:
        with open(config_path) as f:
            config = toml.load(f)
    else:
        config = {}
    logging.info('Registration options are loaded.')

    output_directory = resolve_variable('output_directory', output_directory, config.get('options', None), 'out')
    path_to_transform = resolve_variable('path_to_transform', path_to_transform, config.get('options', None), None)
    if path_to_transform is None:
        print('No transform provided.')
        exit(0)
    # TODO: Load registration_result
    registration_result = None

    moving_histology_section = HistologySection(ref_image=None, ref_mask=None)
    for default_image_path in default_images:
        default_image = DefaultImage.load_from_path(default_image_path)
        moving_histology_section.additional_data.append(default_image)
    for default_annotation_path in default_annotations:
        default_annotation = DefaultImage.load_from_path(default_annotation_path, is_annotation=True)
        moving_histology_section.additional_data.append(default_annotation)
    for tif_image_path in tif_images:
        tif_image = OMETIFFImage.load_from_path(tif_image_path)
        moving_histology_section.additional_data.append(tif_image)
    for tif_annotation_path in tif_annotations:
        tif_annotation = OMETIFFImage.load_and_transform_data(tif_annotation_path, is_annotation=True)
        moving_histology_section.additional_data.append(tif_annotation)
    for pointset_path in pointsets:
        pointset = Pointset.load_from_path(pointset_path)
        moving_histology_section.additional_data.append(pointset)
    for geojson_path in geojsons:
        geojson_data = GeoJsonData.load_from_path(geojson_path)
        moving_histology_section.additional_data.append(geojson_data)
    
    # Setup file structure
    output_directory_registrations = join(output_directory, 'registrations')
    create_if_not_exists(output_directory_registrations)

    output_directory_transformation_data = join(output_directory, 'transformed_data')
    create_if_not_exists(output_directory_transformation_data)

    warped_histology_section = moving_histology_section.apply_transformation(registration_result=registration_result)
    warped_histology_section.to_directory(output_directory_transformation_data)


def groupwise_registration(config_path: str):
    with open(config_path) as f:
        config = toml.load(f)
    if 'gfh_options' in config:
        registration_options = RegistrationOptions.parse_cmdln_dict(config['gfh_options'])
    else:
        registration_options = RegistrationOptions.default_options()
    if 'options' in config:
        # TODO: Should I add collision avoidance?
        output_directory = config['options'].get('output_directory', 'out')
        path_to_greedy = config['options'].get('path_to_greedy', '')
    else:
        output_directory = 'out'
        path_to_greedy = ''

    path_to_greedy = resolve_variable('path_to_greedy', path_to_greedy, config.get('options', None), '')
    registerer = GreedyFHist.load_from_config({'path_to_greedy': path_to_greedy})

    sections = load_sections(config['input'])

    img_mask_list = []
    for section in sections:
        img = section.ref_image.data
        mask = section.ref_mask.data if section.ref_mask is not None else None
        img_mask_list.append((img, mask))
    group_reg, _ = registerer.groupwise_registration(img_mask_list, 
                                                     affine_options=registration_options)

    for idx, section in enumerate(sections[:-1]):
        # Replace this with section id.
        section_output_directory = join(output_directory, f'section{idx}')
        create_if_not_exists(section_output_directory)
        section_output_directory_transform = join(section_output_directory, 'registration')
        create_if_not_exists(section_output_directory_transform)
        transform = group_reg.get_transforms(idx)
        transform.to_file(section_output_directory_transform)
        warped_section = section.apply_transformation(transform, registerer)
        section_output_directory_data = join(section_output_directory, 'transformed_data')
        create_if_not_exists(section_output_directory_data)
        warped_section.to_directory(section_output_directory_data)

    section_output_directory = join(output_directory, f'section{len(sections)-1}')
    create_if_not_exists(section_output_directory)
    section_output_directory_data = join(section_output_directory, 'transformed_data')
    create_if_not_exists(section_output_directory_data)
    sections[-1].to_directory(section_output_directory_data)
    