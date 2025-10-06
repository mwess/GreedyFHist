import ast
from collections import OrderedDict
from dataclasses import dataclass
import logging
from os.path import join
from os import PathLike
import re
from typing import Any

import cv2
import toml

from greedyfhist.utils.io import create_if_not_exists, write_to_ometiffile
from greedyfhist.registration.greedy_f_hist import GreedyFHist, RegistrationTransforms, RegistrationResult
from greedyfhist.options.options import RegistrationOptions
from greedyfhist.data_types import (
    Pointset, 
    GeoJsonData, 
    HistologySection, 
    Image,
    InterpolationConfig, 
    INTERPOLATION_TYPE,
    INTERPOLATION_LIST_TYPE,
    INTERPOLATION_DICT_TYPE,
    INTERPOLATION_LIST2_TYPE,
    INTERPOLATION_DICT2_TYPE
)


logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.WARN)


def convert_string_to_bool(val: str) -> bool:
    if val in ['True', 'T', 'true', 't', 'TRUE']:
        return True
    elif val in ['False', 'F', 'false', 'f', 'FALSE']:
        return False
    else:
        raise ValueError(f'Input argument {val} could not be converted to bool.')

        
def convert_non_string_int_types_to_str(cont: int | str | list | dict) -> int | str | list | dict:
    if isinstance(cont, list):
        ret = []
        for elem in cont:
            elem_conv = convert_non_string_int_types_to_str(elem)
            ret.append(elem_conv)
        return ret
    elif isinstance(cont, dict):
        ret = {}
        for key in cont:
            value = cont[key]
            new_key = convert_non_string_int_types_to_str(key)
            new_value = convert_non_string_int_types_to_str(value)
            ret[new_key] = new_value
        return ret
    elif not (isinstance(cont, str) or isinstance(cont, int)):
        return str(cont)
    else:
        return cont

        
def convert_string_to_interpolation_dict(val: str,
                                         convert_non_string_int_types: bool = True) -> dict | str | int | list | None:
    if not val:
        return {}
    try:
        cont = ast.literal_eval(val)
        if convert_non_string_int_types:
            cont = convert_non_string_int_types_to_str(cont)
        return cont
    except SyntaxError:
        return val


def try_to_int(s: str | int) -> int | str:
    try:
        return int(s) # type: ignore
    except ValueError:
        return s


# IO config classes
@dataclass
class ImageConfig:

    path: str | PathLike | None = None
    main_image: str | int = 0 
    main_channel: str | int = 0 
    order: str | None = None
    interpolate_default: INTERPOLATION_TYPE = 'LINEAR'
    interpolate_default_per_scene: INTERPOLATION_LIST_TYPE | INTERPOLATION_DICT_TYPE | None = None
    interpolate_default_per_channel: INTERPOLATION_LIST2_TYPE | INTERPOLATION_DICT2_TYPE | None = None
    suppress_dictionary_conversion: bool = False
    force_parameters_as_string: bool = False
    # TODO: Implement reader variable for bioio. Maybe can be done with some enum?
    # reader ...

    def __post_init__(self):
        if isinstance(self.suppress_dictionary_conversion, str):
            self.suppress_dictionary_conversion = convert_string_to_bool(self.suppress_dictionary_conversion)
        if isinstance(self.force_parameters_as_string, str):
            self.force_parameters_as_string = convert_string_to_bool(self.force_parameters_as_string)
        if self.interpolate_default_per_scene and not self.suppress_dictionary_conversion:
            self.interpolate_default_per_scene = convert_string_to_interpolation_dict(self.interpolate_default_per_scene) # type: ignore
        if self.interpolate_default_per_channel and not self.suppress_dictionary_conversion:
            self.interpolate_default_per_channel = convert_string_to_interpolation_dict(self.interpolate_default_per_channel) # type: ignore
        if not self.force_parameters_as_string:
            self.main_image = try_to_int(self.main_image)
            self.main_channel = try_to_int(self.main_channel)

    def __repr__(self):
        repr_str = "ImageConfig:"
        repr_str += f' path={self.path}'
        repr_str += f' main_image={self.main_image}'
        repr_str += f' main_channel={self.main_channel}'
        repr_str += f' order={self.order}'
        repr_str += f' interpolate_default={self.interpolate_default}'
        repr_str += f' interpolate_default_per_image={self.interpolate_default_per_scene}'
        repr_str += f' interpolate_default_per_channel={self.interpolate_default_per_channel}'
        repr_str += f' suppress_dictionary_conversion={self.suppress_dictionary_conversion}'
        repr_str += f' force_parameters_as_string={self.force_parameters_as_string}'
        return repr_str

    def load_image(self) -> Image:
        interpolation_config = InterpolationConfig(
            interpolate_default=self.interpolate_default,
            interpolate_per_scene=self.interpolate_default_per_scene,
            interpolate_per_channel=self.interpolate_default_per_channel
        )
        if not self.path:
            raise ValueError('Image needs a path to be loaded.')
        return Image.load_from_path(
            path = self.path,
            order = self.order,
            main_image = self.main_image,
            main_channel = self.main_channel,
            interpolation_config = interpolation_config
        )

    @staticmethod
    def get_empty_arg_list() -> list:
        return [None, 0, 0, None, 'LINEAR', None, None, False, False]

    @staticmethod
    def get_empty_arg_dict() -> dict:
        od = OrderedDict()
        od['path'] = None
        od['main_image'] = 0
        od['main_channel'] = 0
        od['order'] = None
        od['interpolate_default'] = 'LINEAR'
        od['interpolate_default_per_scene'] = None
        od['interpolate_default_per_channel'] = None
        od['suppress_dictionary_conversion'] = False
        od['force_parameters_as_string'] = False
        return od

    @classmethod
    def parse_from_cmd_str(cls, value: str, requires_image: bool = True) -> 'ImageConfig':
        # args = value.split(',')
        args = re.split(r',(?![^\{\(]*\})(?![^\[\(]*\])', value)
        args = [x.strip() for x in args]
        if not args[0] and requires_image:
            raise ValueError('No image provided.')
        parsed_args = ImageConfig.get_empty_arg_dict()
        if not (1 <= len(args) <= len(parsed_args)):
            raise ValueError(f'Incorrect number of arguments parsed: {len(args)}')
        for idx, (key, arg_str) in enumerate(zip(parsed_args.copy(), args)):
            value = parsed_args[key]
            if '=' in arg_str:
                arg_key, arg_value = arg_str.split('=', maxsplit=1)
                parsed_args[arg_key] = arg_value
            else:
                parsed_args[key] = arg_str
        return cls(**parsed_args)


@dataclass
class PointsetConfig:
    
    path: str | PathLike
    x_axis: str | int = 0
    y_axis: str | int = 1
    header: int | None = None
    force_parameters_as_string: bool = False

    def __post_init__(self):
        if isinstance(self.force_parameters_as_string, str):
            self.force_parameters_as_string = convert_string_to_bool(self.force_parameters_as_string)
        if not self.force_parameters_as_string:
            self.x_axis = try_to_int(self.x_axis)
            self.y_axis = try_to_int(self.y_axis)
        if self.header is not None:
            self.header = try_to_int(self.header) # type: ignore

    def load_pointset(self) -> Pointset:
        ps = Pointset.load_from_path(
            path=self.path,
            x_axis=self.x_axis,
            y_axis=self.y_axis,
            header=self.header
        )
        return ps

    @staticmethod
    def get_empty_arg_dict() -> dict:
        od = OrderedDict()
        od['path'] = None
        od['x_axis'] = 0
        od['y_axis'] = 0
        od['header'] = None
        od['force_parameters_as_string'] = False
        return od

    @classmethod
    def parse_from_cmd_str(cls, value: str) -> 'PointsetConfig':
        args = value.split(',')
        parsed_args = PointsetConfig.get_empty_arg_dict()
        if not (1 <= len(args) <= len(parsed_args)):
            raise ValueError(f'Incorrect number of arguments parsed: {len(args)}')
        for idx, (key, arg_str) in enumerate(zip(parsed_args.copy(), args)):
            value = parsed_args[key]
            if '=' in arg_str:
                arg_key, arg_value = arg_str.split('=', maxsplit=1)
                parsed_args[arg_key] = arg_value
            else:
                parsed_args[key] = arg_str
        return cls(**parsed_args)


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
        return None
    choice2 = config.get(selector, None)
    if choice2 is None:
        return fallback_value
    return choice2

    
def load_image_from_config(config: dict) -> Image:
    """Load image based on config.

    Args:
        config (Dict):

    Returns:
        Union[DefaultImage, OMETIFFIMage]: 
    """
    return Image.load_from_config(config)


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
        return Image.load_from_config(config)
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


def load_base_histology_section(image_config: ImageConfig | None = None,
                                mask_config: ImageConfig | None = None) -> 'HistologySection':
    """Loads histology section from filepath. Optionally an image mask can be parsed along.

    Args:
        image_path (str | None, optional): Defaults to None.
        mask_path (str | None, optional): Defaults to None.

    Returns:
        HistologySection: 
    """
    if image_config is not None:
        image = image_config.load_image()
    else:
        image = None
    if mask_config is not None:
        if not mask_config.path and image_config is not None:
            mask_config.path = image_config.path
        mask = mask_config.load_image()
    else:
        mask = None
    histology_section = HistologySection(ref_image=image,
                                         ref_mask=mask)
    return histology_section


def register(moving_image_config: ImageConfig,
              fixed_image_config: ImageConfig,
              output_directory: str | PathLike,
              moving_mask_config: ImageConfig | None = None,
              fixed_mask_config: ImageConfig | None = None,
              path_to_greedy: str | PathLike | None = None,
              use_docker_executable: bool = False,
              images: list[ImageConfig] | None = None,
              pointsets: list[PointsetConfig] | None = None,
              geojsons: list[str | PathLike] | None = None,
              config_path: str | PathLike | None = None):
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
        use_docker_executable (bool, None): Defaults to None.
        config_path (Optional[str], optional): Defaults to None.
        images (Optional[List[str]], optional): Defaults to None.
        annotations (Optional[List[str]], optional): Defaults to None.
        pointsets (Optional[List[str]], optional): Defaults to None.
        geojsons (Optional[List[str]], optional): Defaults to None.
    """
    # Parse input parameters correctly.
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
    # TODO: If anything is provided in the config, take it from there.
    # if all_paths_are_none([moving_image_config, fixed_image_config, moving_mask_config, fixed_mask_config]):
    #     moving_image_config, fixed_image_config, moving_mask_config, fixed_mask_config = get_paths_from_config(config.get('input', None))
    if moving_image_config is None and fixed_image_config is None:
        raise Exception('No moving and fixed image path provided!')

    output_directory = resolve_variable('output_directory', output_directory, config.get('options', None), 'out') # type: ignore
    path_to_greedy = resolve_variable('path_to_greedy', path_to_greedy, config.get('options', None), '')
    save_transform_to_file = resolve_variable('save_transform_to_file', None, config.get('options', None), True)

    # Load image data.
    moving_histology_section = load_base_histology_section(
        image_config=moving_image_config,
        mask_config=moving_mask_config
    )

    # Setup file structure
    output_directory_registrations = join(output_directory, 'transformation')
    create_if_not_exists(output_directory_registrations)

    fixed_image = fixed_image_config.load_image()
    if fixed_mask_config is not None:
        fixed_mask = fixed_mask_config.load_image()
    else:
        fixed_mask = None

    logging.info('Loaded images. Starting registration.')
    logging.info(f'Registration options: {registration_options}')
    registerer = GreedyFHist(path_to_greedy=str(path_to_greedy), use_docker_container=use_docker_executable)

    moving_mask = moving_histology_section.ref_mask.data.squeeze() if moving_histology_section.ref_mask is not None else None
    fixed_mask = fixed_mask.data.squeeze() if fixed_mask is not None else None
    registration_result = registerer.register(
        moving_histology_section.ref_image.data.squeeze(), # type: ignore
        fixed_image.data.squeeze(),
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

    #This should be made more efficient. Its not necessary to load everything before transformation.
    # Each image should be loaded, transformed, and stored before the next one is processed.
    if images is None:
        images = []
    if pointsets is None:
        pointsets = []
    if geojsons is None:
        geojsons = []
    for image_config in images:
        image = image_config.load_image()
        moving_histology_section.additional_data.append(image)
    for pointset_config in pointsets:
        pointset = pointset_config.load_pointset() 
        moving_histology_section.additional_data.append(pointset)
    for geojson_path in geojsons:
        geojson_data = GeoJsonData.load_from_path(geojson_path)
        moving_histology_section.additional_data.append(geojson_data)

    warped_histology_section = moving_histology_section.apply_transformation(
        registration_transforms=registration_result.registration,
        registerer=registerer)
    warped_histology_section.to_directory(output_directory_transformation_data)
    output_directory_transformation_data_prep = join(output_directory_transformation_data, 'preprocessing_data')
    create_if_not_exists(output_directory_transformation_data_prep)
    # Try writing masks to file.
    try:
        moving_preprocessing_mask = registration_result.registration.reg_params.moving_preprocessing_params['moving_img_mask'] # type: ignore
        path = join(output_directory_transformation_data_prep, 'moving_mask.png')
        cv2.imwrite(path, moving_preprocessing_mask)
        fixed_preprocessing_mask = registration_result.registration.reg_params.fixed_preprocessing_params['fixed_img_mask'] # type: ignore
        path = join(output_directory_transformation_data_prep, 'fixed_mask.png')
        cv2.imwrite(path, fixed_preprocessing_mask)
    except Exception:
        print('Masks could not be stored due to error.')

  
def apply_transformation(
             output_directory: str | None = None,
             config_path: str | None = None,
             path_to_transform: str | None = None,
             images: list[ImageConfig] | None = None,
             pointsets: list[PointsetConfig] | None = None,
             geojsons: list[str | PathLike] | None = None):
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
    pass
    if images is None:
        images = []
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
    for image_config in images:
        image = image_config.load_image()
        moving_histology_section.additional_data.append(image)
    for pointset_config in pointsets:
        pointset = pointset_config.load_pointset()
        moving_histology_section.additional_data.append(pointset)
    for geojson_path in geojsons:
        geojson_data = GeoJsonData.load_from_path(geojson_path)
        moving_histology_section.additional_data.append(geojson_data)
    
    # Setup file structure
    output_directory_registrations = join(output_directory, 'registrations') # type: ignore
    create_if_not_exists(output_directory_registrations)

    output_directory_transformation_data = join(output_directory, 'transformed_data') # type: ignore
    create_if_not_exists(output_directory_transformation_data)

    warped_histology_section = moving_histology_section.apply_transformation(
        registration_transforms=registration_result)
    warped_histology_section.to_directory(output_directory_transformation_data)


# def groupwise_registration(moving_images_config: list[ImageConfig],
#                            fixed_image_config: ImageConfig,
#                            output_directory: str,
#                            moving_masks_config: list[ImageConfig],
#                            fixed_mask_config: ImageConfig | None,
#                            path_to_greedy: str | PathLike | None,
#                            use_docker_executable: bool,
#                            config):
def groupwise_registration(config):
    """Performs groupwise registration based on config.

    Args:
        config_path (str):
    """
    pass
    if config is not None:
        with open(config) as f:
            config = toml.load(f)
    else:
        config = {}
    reg_opts = config.get('gfh_options', {})
    options = RegistrationOptions.parse_cmdln_dict(reg_opts)

    path_to_greedy: str = resolve_variable('path_to_greedy', None, config.get('options', None), '') # type: ignore
    use_docker_executable: bool = resolve_variable('use_docker_container', None, config.get('options', None), False) # type: ignore
    save_transform_to_file: bool = resolve_variable('save_transform_to_file', None, config.get('options', None), True)  #type: ignore      
    output_directory: str = resolve_variable('output_directory', None, config.get('options', None), 'out') # type: ignore
    registerer = GreedyFHist(path_to_greedy=path_to_greedy, # type: ignore
                             use_docker_container=use_docker_executable)

    sections = load_sections(config['input'])

    img_mask_list = []
    for section in sections:
        img = section.ref_image.data.squeeze() # type: ignore
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