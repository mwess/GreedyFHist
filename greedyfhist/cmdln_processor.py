import logging
from os.path import join
from typing import Any

import cv2
import toml

from greedyfhist.utils.io import create_if_not_exists, write_to_ometiffile
from greedyfhist.registration.greedy_f_hist import GreedyFHist, RegistrationTransforms, RegistrationResult
from greedyfhist.options.options import RegistrationOptions
from greedyfhist.data_types import Pointset, GeoJsonData, HistologySection, Image


logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.WARN)


def all_paths_are_none(path_list: list[str]) -> bool:
    """Checks whether all paths are None.

    Args:
        path_list (List[str]): List of file paths.

    Returns:
        bool: 
    """
    return all(map(lambda x: x is None, path_list))


def get_paths_from_config(config: dict | None = None) -> tuple[str | None, str | None, str | None, str | None]:
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
    

def resolve_variable(selector: str, 
                    choice1: Any | None = None, 
                    config: Any | None = None,
                    fallback_value: Any | None = None) -> Any | None:
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
        return fallback_value
    choice2 = config.get(selector, None)
    if choice2 is None or choice2 == '':
        return fallback_value
    return choice2

    
def load_image_from_config(config: dict) -> Image:
    """Load image based on config.

    Args:
        config (Dict):

    Returns:
        Union[DefaultImage, OMETIFFIMage]: 
    """
    return Image.load_data_from_config(config)


def guess_type(path: str) -> str:
    """Guesses which intrinsic GreedyFHist datatype fits best.

    Args:
        path (str): 

    Returns:
        str:
    """
    if path.endswith('geojson'):
        return 'geojson'
    if path.endswith('csv'):
        return 'pointset'
    if path.endswith('tiff') or path.endswith('tif'):
        return 'tiff'
    return 'default'


def load_data_from_config(config: dict) -> Image | Pointset | GeoJsonData:
    """Loads datatypes from config.

    Args:
        config (dict):

    Returns:
        Image | Pointset | GeoJsonData:
    """
    type_ = config.get('type', None)
    if type_ is None:
        type_ = guess_type(config['path'])
    if type_ == 'geojson':
        return GeoJsonData.load_data(config)
    elif type_ == 'pointset':
        return Pointset.load_data(config)
    else:
        return Image.load_data_from_config(config)
    # Throw an error message otherwise.


def load_sections(section_configs: dict) -> list[HistologySection]:
    """Loads multiple histology sections from config.

    Args:
        section_configs (dict):

    Returns:
        list[HistologySection]:
    """
    sections = []
    sorted_keys = sorted(section_configs.keys(), key=lambda x: int(x.replace('section', '')))
    for key in sorted_keys:
        section_config = section_configs[key]
        histology_section = load_histology_section_from_config(section_config)
        sections.append(histology_section)
    return sections


def load_histology_section_from_config(config: dict) -> HistologySection:
    """Loads one histology section from config.

    Args:
        config (dict): 

    Returns:
        HistologySection: 
    """
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


def load_base_histology_section(image_path: str | None = None,
                                mask_path: str | None = None) -> 'HistologySection':
    """Loads histology section from filepath. Optionally an image mask can be parsed along.

    Args:
        image_path (str | None, optional): Defaults to None.
        mask_path (str | None, optional): Defaults to None.

    Returns:
        HistologySection: 
    """
    if image_path is not None:
        image = Image.load_from_path(image_path)
    else:
        image = None
    if mask_path is not None:
        mask = Image.load_from_path(mask_path, True)
    else:
        mask = None
    histology_section = HistologySection(ref_image=image,
                                         ref_mask=mask)
    return histology_section


def register(moving_image_path: str | None = None,
             fixed_image_path: str | None = None,
             output_directory: str | None = None,
             moving_mask_path: str | None = None,
             fixed_mask_path: str | None = None,
             path_to_greedy: str | None = None,
             use_docker_executable: bool | None = None,
             config_path: str | None = None,
             images: list[str] | None = None,
             annotations: list[str] | None = None,
             pointsets: list[str] | None = None,
             geojsons: list[str] | None = None):
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
    # Parse input parameters correctly.
    if images is None:
        images = []
    if annotations is None:
        annotations = []
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
    if use_docker_executable:
        registration_options.use_docker_container = use_docker_executable
    logging.info('Registration options are loaded.')
    if all_paths_are_none([moving_image_path, fixed_image_path, moving_mask_path, fixed_mask_path]):
        moving_image_path, fixed_image_path, moving_mask_path, fixed_mask_path = get_paths_from_config(config.get('input', None))
    if moving_image_path is None and fixed_image_path is None:
        raise Exception('No moving and fixed image path provided!')

    output_directory = resolve_variable('output_directory', output_directory, config.get('options', None), 'out')
    path_to_greedy = resolve_variable('path_to_greedy', path_to_greedy, config.get('options', None), 'greedy')
    save_transform_to_file = resolve_variable('save_transform_to_file', None, config.get('options', None), True)

    # Load image data.
    moving_histology_section = load_base_histology_section(
        image_path=moving_image_path,
        mask_path=moving_mask_path
    )
    
    for image_path in images:
        image = Image.load_from_path(image_path)
        moving_histology_section.additional_data.append(image)
    for annotation_path in annotations:
        annotation = Image.load_from_path(annotation_path, True)
        moving_histology_section.additional_data.append(annotation)
    for pointset_path in pointsets:
        pointset = Pointset.load_from_path(pointset_path)
        moving_histology_section.additional_data.append(pointset)
    for geojson_path in geojsons:
        geojson_data = GeoJsonData.load_from_path(geojson_path)
        moving_histology_section.additional_data.append(geojson_data)


    # Setup file structure
    output_directory_registrations = join(output_directory, 'transformation')
    create_if_not_exists(output_directory_registrations)

    fixed_image = Image.load_from_path(fixed_image_path)
    fixed_mask = Image.load_from_path(fixed_mask_path, is_annotation=True) if fixed_mask_path is not None else None

    logging.info('Loaded images. Starting registration.')
    logging.info(f'Registration options: {registration_options}')
    registerer = GreedyFHist(path_to_greedy=path_to_greedy, use_docker_container=use_docker_executable)
    
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
        registration_result.to_directory(output_directory_registrations)
        logging.info('Registration saved.')

    
    output_directory_transformation_data = join(output_directory, 'transformed_data')
    create_if_not_exists(output_directory_transformation_data)

    warped_histology_section = moving_histology_section.apply_transformation(registration_result=registration_result.registration,
                                                                             registerer=registerer)
    warped_histology_section.to_directory(output_directory_transformation_data)
    output_directory_transformation_data_prep = join(output_directory_transformation_data, 'preprocessing_data')
    create_if_not_exists(output_directory_transformation_data_prep)
    # Try writing masks to file.
    try:
        moving_preprocessing_mask = registration_result.registration.reg_params.moving_preprocessing_params['moving_img_mask']
        path = join(output_directory_transformation_data_prep, 'moving_mask.png')
        cv2.imwrite(path, moving_preprocessing_mask)
        fixed_preprocessing_mask = registration_result.registration.reg_params.fixed_preprocessing_params['fixed_img_mask']
        path = join(output_directory_transformation_data_prep, 'fixed_mask.png')
        cv2.imwrite(path, fixed_preprocessing_mask)
    except Exception:
        print('Masks could not be stored due to error.')


  
def apply_transformation(
             output_directory: str | None = None,
             config_path: str | None = None,
             path_to_transform: str | None = None,
             images: list[str] | None = None,
             annotations: list[str] | None = None,
             pointsets: list[str] | None = None,
             geojsons: list[str] | None = None):
    """Transforms provided data from moving to fixed image space by
    load a computed transformation. If registration_result
    is None, a registration_result is read from file. Registration_result
    is first applied to data provided as command line arguments (images,
    annotations, pointsets, geojsons). Then data declared in the config 
    file is transformed. All transformed data is stored in 
    'output_directory'/transformed_data.


    Args:
        output_directory (str | None, optional): _description_. Defaults to None.
        config_path (str | None, optional): _description_. Defaults to None.
        path_to_transform (str | None, optional): _description_. Defaults to None.
        images (list[str] | None, optional): _description_. Defaults to None.
        annotations (list[str] | None, optional): _description_. Defaults to None.
        pointsets (list[str] | None, optional): _description_. Defaults to None.
        geojsons (list[str] | None, optional): _description_. Defaults to None.
    """
    if images is None:
        images = []
    if annotations is None:
        annotations = []
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
    registration_result = RegistrationTransforms.load(path_to_transform)

    moving_histology_section = HistologySection(ref_image=None, ref_mask=None)
    for image_path in images:
        image = Image.load_from_path(image_path)
        moving_histology_section.additional_data.append(image)
    for annotation_path in annotations:
        annotation = Image.load_from_path(annotation_path, is_annotation=True)
        moving_histology_section.additional_data.append(annotation)
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
    """Performs groupwise registration based on config.

    Args:
        config_path (str):
    """
    if config_path is not None:
        with open(config_path) as f:
            config = toml.load(f)
    else:
        config = {}
    reg_opts = config.get('gfh_options', {})
    options = RegistrationOptions.parse_cmdln_dict(reg_opts)

    path_to_greedy = resolve_variable('path_to_greedy', None, config.get('options', None), 'greedy')
    use_docker_executable = resolve_variable('use_docker_container', None, config.get('options', None), False)
    save_transform_to_file = resolve_variable('save_transform_to_file', None, config.get('options', None), True)        
    output_directory = resolve_variable('output_directory', None, config.get('options', None), 'out')
    registerer = GreedyFHist(path_to_greedy=path_to_greedy, use_docker_container=use_docker_executable)

    sections = load_sections(config['input'])

    img_mask_list = []
    for section in sections:
        img = section.ref_image.data
        mask = section.ref_mask.data if section.ref_mask is not None else None
        img_mask_list.append((img, mask))
    group_reg, _ = registerer.groupwise_registration(img_mask_list,
                                                     options=options)
    for idx, section in enumerate(sections[:-1]):
        # Replace this with section id.
        section_output_directory = join(output_directory, f'section{idx}')
        create_if_not_exists(section_output_directory)
        transform = group_reg.get_transforms(idx)
        if save_transform_to_file:
            section_output_directory_transform = join(section_output_directory, 'transformations')
            create_if_not_exists(section_output_directory_transform)
            transform.to_directory(section_output_directory_transform)
        warped_section = section.apply_transformation(transform.registration, registerer)
        section_output_directory_data = join(section_output_directory, 'transformed_data')
        create_if_not_exists(section_output_directory_data)
        warped_section.to_directory(section_output_directory_data)

    section_output_directory = join(output_directory, f'section{len(sections)-1}')
    create_if_not_exists(section_output_directory)
    section_output_directory_data = join(section_output_directory, 'transformed_data')
    create_if_not_exists(section_output_directory_data)
    sections[-1].to_directory(section_output_directory_data)        
    if save_transform_to_file:
        output_directory_group_transforms = join(output_directory, 'group_transforms')
        create_if_not_exists(output_directory_group_transforms)
        group_reg.to_file(output_directory_group_transforms)    