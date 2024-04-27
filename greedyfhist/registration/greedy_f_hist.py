from collections import OrderedDict
import copy
from dataclasses import dataclass
import json
import os
from os.path import join, exists
import shutil
import subprocess
from typing import Any, Callable, Dict, List, Optional, Tuple

import geojson
import numpy
import numpy as np
import pandas
import pandas as pd
import SimpleITK as sitk

from greedyfhist.utils.io import create_if_not_exists, read_simple_vtk, write_coordinates_as_vtk, write_mat_to_file
from greedyfhist.utils.image import (
    call_command,
    rescale_affine,
    rescale_warp,
    denoise_image,
    apply_mask,
    com_affine_matrix,
    pad_image,
    resample_image_with_gaussian,
    pad_asym,
    get_symmetric_padding,
    remove_padding,
    cropping,
    add_cropped_region,
    resample_by_factor,
    resize_image
)
from greedyfhist.utils.utils import build_cmd_string, scale_table
from greedyfhist.utils.geojson_utils import geojson_2_table, convert_table_2_geo_json
from greedyfhist.segmentation.segmenation import load_yolo_segmentation
from greedyfhist.options import Options, GreedyOptions


def deformable_registration(path_to_greedy:str,
                            path_fixed_image:str,
                            path_moving_image:str,
                            options: GreedyOptions,
                            output_warp:Optional[str]=None,
                            output_inv_warp:Optional[str]=None,
                            affine_pre_transform=None,
                            ):
    """Calls the deformable registration command of greedy.

    Args:
        path_to_greedy (str): 
        path_fixed_image (str): 
        path_moving_image (str): 
        options (GreedyOptions): Contains options to pass to greedy.
        output_warp (str, optional): Defaults to None.
        output_inv_warp (str, optional): Defaults to None.
        affine_pre_transform (_type_, optional): Contains path to affine_pre_transform. Necessary if ia is ia-com-init. Defaults to None.

    Returns:
        _type_: Return of command line execution.
    """
    cost_fun_params = options.cost_function
    if options.cost_function == 'ncc' or options.cost_function == 'wncc':
        cost_fun_params += f' {options.kernel_size}x{options.kernel_size}'
    def_args = {}
    def_args['-it'] = affine_pre_transform
    def_args['-d'] = options.dim
    # def_args['-m'] = f'NCC {kernel}x{kernel}'
    def_args['-m'] = cost_fun_params
    def_args['-i'] = [path_fixed_image, path_moving_image]
    pyramid_iterations = 'x'.join([str(x) for x in options.deformable_pyramid_iterations])
    def_args['-n'] = pyramid_iterations
    # def_args['-n'] = f'{options.pyramid_iterations[0]}x{options.pyramid_iterations[1]}x{options.pyramid_iterations[2]}'
    # def_args['-threads'] = '32'
    def_args['-threads'] = options.n_threads
    def_args['-s'] = [f'{options.s1}vox', f'{options.s2}vox']
    def_args['-o'] = output_warp
    def_args['-oinv'] = output_inv_warp
    if options.use_sv:
        def_args['-sv'] = ''
    elif options.use_svlb:
        def_args['-svlb'] = ''

    def_cmd = build_cmd_string(path_to_greedy, def_args)
    def_ret = call_command(def_cmd)
    return def_ret


def affine_registration(path_to_greedy: str,
                        path_to_fixed_image: str,
                        path_to_moving_image: str,
                        path_output: str,
                        offset:int,
                        ia:str,
                        options: GreedyOptions,
                        ):
    """Calls greedy's affine registration function.

    Args:
        path_to_greedy (str): _description_
        path_to_fixed_image (str): _description_
        path_to_moving_image (str): _description_
        path_output (str): _description_
        offset (int): _description_
        ia (str): _description_
        options (GreedyOptions): _description_

    Returns:
        _type_: Return of command line execution.
    """
    cost_fun_params = options.cost_function
    if options.cost_function == 'ncc' or options.cost_function == 'wncc':
        cost_fun_params += f' {options.kernel_size}x{options.kernel_size}'
    aff_rgs = {}
    aff_rgs['-d'] = options.dim
    aff_rgs['-i'] = [path_to_fixed_image, path_to_moving_image]
    aff_rgs['-o'] = path_output
    aff_rgs['-m'] = cost_fun_params
    pyramid_iterations = 'x'.join([str(x) for x in options.affine_pyramid_iterations])
    # aff_rgs['-n'] = f'{options.pyramid_iterations[0]}x{options.pyramid_iterations[1]}x{options.pyramid_iterations[2]}'
    aff_rgs['-n'] = pyramid_iterations
    aff_rgs['-threads'] = options.n_threads
    aff_rgs['-dof'] = '12'
    aff_rgs['-search'] = f'{options.iteration_rigid} 180 {offset}'.split()  # Replaced 360 with any for rotation parameter
    aff_rgs['-gm-trim'] = f'{options.kernel_size}x{options.kernel_size}'
    aff_rgs['-a'] = ''  # Doesnt get param how to parse?
    aff_rgs[ia[0]] = ia[1]

    aff_cmd = build_cmd_string(path_to_greedy, aff_rgs)
    aff_ret = call_command(aff_cmd)
    return aff_ret


# Warp functions

def warp_image_data(fixed_image_path:str,
                    moving_image_path:str,
                    output_image_path:str,
                    transforms,
                    path_to_greedy:str,
                    path_to_c2d:str,
                    temp_dir:str,
                    interpolation_mode:str='LINEAR'):
    """Warps image data using greedy.

    Args:
        fixed_image_path (str): _description_
        moving_image_path (str): _description_
        output_image_path (str): _description_
        transforms (_type_): _description_
        path_to_greedy (str): _description_
        path_to_c2d (str): _description_
        temp_dir (str): _description_
        interpolation_mode (str, optional): _description_. Defaults to 'LINEAR'.

    Returns:
        _type_: _description_
    """
    # TODO: Is this command just copying images into a new folder?
    fixed_fname = os.path.basename(fixed_image_path)
    path_new_target = os.path.join(temp_dir, fixed_fname + '_new_target.nii.gz')
    cmd = f"""{path_to_c2d} -mcs '{fixed_image_path}' -foreach -orient LP -spacing 1x1mm -origin 0x0mm -endfor -omc '{path_new_target}'"""
    fixed_pre_ret = call_command(cmd)

    moving_fname = os.path.basename(moving_image_path)
    path_new_source = os.path.join(temp_dir, moving_fname + '_new_source.nii.gz')
    cmd = f"""{path_to_c2d} -mcs '{moving_image_path}' -foreach -orient LP -spacing 1x1mm -origin 0x0mm -endfor -omc '{path_new_source}'"""
    moving_pre_ret = call_command(cmd)

    big_reslice_args = {}
    big_reslice_args['-r'] = transforms
    big_reslice_args['-d'] = '2'
    big_reslice_args['-rf'] = path_new_target
    big_reslice_args['-rm'] = [path_new_source, output_image_path]
    big_reslice_args['-ri'] = [interpolation_mode]

    big_reslice_cmd = build_cmd_string(path_to_greedy, big_reslice_args)
    big_reslice_ret = call_command(big_reslice_cmd)
    return [fixed_pre_ret, moving_pre_ret, big_reslice_ret]


def map_infs_back(warped_landmarks: pandas.DataFrame, source_landmarks: pandas.DataFrame) -> pandas.DataFrame:
    """Maps back any inf values that had to be removed prior to warping.

    Args:
        warped_landmarks (pandas.DataFrame):
        source_landmarks (pandas.DataFrame):

    Returns:
        pandas.DataFrame: 
    """
    infs = (source_landmarks.x_small == np.inf) | (source_landmarks.y_small == np.inf)
    infs = infs.to_numpy()
    warped_landmarks.loc[infs, 'x'] = np.inf
    warped_landmarks.loc[infs, 'y'] = np.inf
    # warped_landmarks.x.loc[infs] = np.inf
    # warped_landmarks.y.loc[infs] = np.inf
    return warped_landmarks


def process_and_warp_coordinates(path_to_greedy: str,
                                 path_coordinates: str,
                                 output_path: str,
                                 path_temp: str,
                                 paths_to_inv_transforms: List[str],
                                 path_small_ref_img: str,
                                 width_downscale: float,
                                 height_downscale: float,
                                 moving_padding: Tuple[int, int, int, int],
                                 fixed_padding: Tuple[int, int, int, int],
                                 cropped_params_mov: Tuple[int, int, int, int],
                                 cropped_params_fix: Tuple[int, int, int, int],
                                 additional_args: Optional[Dict[str, Any]]=None) -> 'PointcloudTransformationResult':
    """Process pointset data through various image operations and warps coordinates using greedy.

    Args:
        path_to_greedy (str): _description_
        path_coordinates (str): _description_
        output_path (str): _description_
        path_temp (str): _description_
        paths_to_inv_transforms (List[str]): _description_
        path_small_ref_img (str): _description_
        width_downscale (float): _description_
        height_downscale (float): _description_
        moving_padding (Tuple[int, int, int, int]): _description_
        fixed_padding (Tuple[int, int, int, int]): _description_
        cropped_params_mov (Tuple[int, int, int, int]): _description_
        cropped_params_fix (Tuple[int, int, int, int]): _description_
        additional_args (Optional[Dict[str, Any]], optional): _description_. Defaults to None.

    Returns:
        PointcloudTransformationResult: _description_
    """
    # Apply transformation to landmarks
    if additional_args is None:
        additional_args = {}
    path_small_landmarks = os.path.join(path_temp, 'lm_small_source.vtk')
    df_lm = pd.read_csv(path_coordinates)
    df_lm.x -= cropped_params_mov[2]
    df_lm.y -= cropped_params_mov[0]
    df_lm['x'] = df_lm['x'] + moving_padding[0]
    df_lm['y'] = df_lm['y'] + moving_padding[2]
    # First scale down
    df_lm['x_small'] = df_lm.x * 1.0 * width_downscale - 0.5
    df_lm['y_small'] = df_lm.y * 1.0 * height_downscale - 0.5

    write_coordinates_as_vtk(df_lm[['x_small', 'y_small']].to_numpy(), path_small_landmarks)

    path_small_warped_landmarks = os.path.join(path_temp, 'lm_small_source_warped.vtk')
    transform_params = {}
    transform_params['width_downscale'] = width_downscale
    transform_params['height_downscale'] = height_downscale

    # Second, warp.
    # TODO: Can this be done with big warp as well???
    # TODO: And why do I need interpolation?

    interpolation_mode = additional_args.get('interpolation_mode', 'NN')

    lm_args = {}
    lm_args['-rs'] = [path_small_landmarks, path_small_warped_landmarks]
    lm_args['-r'] = paths_to_inv_transforms
    lm_args['-d'] = '2'
    lm_args['-rf'] = path_small_ref_img
    lm_args['-ri'] = interpolation_mode

    transform_params['interpolation_mode'] = interpolation_mode

    lm_cmd = build_cmd_string(path_to_greedy, lm_args)
    lm_ret = call_command(lm_cmd)
    cmdl_returns = [lm_ret]
    # Third, scale up
    small_warped_lm = read_simple_vtk(path_small_warped_landmarks)
    df_small_warped_lm = pd.DataFrame(small_warped_lm)
    df_small_warped_lm.rename(columns={0: 'small_x', 1: 'small_y'}, inplace=True)
    df_small_warped_lm['x'] = (df_small_warped_lm.small_x + 0.5) * (1 / width_downscale)
    df_small_warped_lm['y'] = (df_small_warped_lm.small_y + 0.5) * (1 / height_downscale)
    df_small_warped_lm.drop(columns=['small_x', 'small_y'], inplace=True)
    df_small_warped_lm = map_infs_back(df_small_warped_lm, df_lm)
    # df_small_warped_lm.to_csv(output_path, index=False)
    transform_params['width_upscale'] = 1 / width_downscale
    transform_params['height_upscale'] = 1 / height_downscale
    df_small_warped_lm['x'] = df_small_warped_lm['x'] - fixed_padding[0]
    df_small_warped_lm['y'] = df_small_warped_lm['y'] - fixed_padding[2]

    df_small_warped_lm['x'] = df_small_warped_lm['x'] + cropped_params_fix[2]
    df_small_warped_lm['y'] = df_small_warped_lm['y'] + cropped_params_fix[0]

    transformation_result = PointcloudTransformationResult(
        pointcloud=df_small_warped_lm,
        pointcloud_path=output_path,
        cmdl_log=cmdl_returns,
        transform_params=transform_params
    )

    return transformation_result


@dataclass
class MultiRegResult:

    reg_results: Any# TODO: Fix this!!#OrderedDict[Any, Any]#OrderedDict[Any, 'RegResult']

    # TODO: Implement get partial stepwise registration
    def get_partial_step_transform(self, src_idx, dst_idx):
        # Gets partial transform from src_idx to dst_idx.
        # TODO: Would probably be better to not rely on indexing and implement something a bit more robuster.
        partial_transforms = [self.reg_results[idx] for idx in range(src_idx, dst_idx)]
        return MultiRegResult(partial_transforms)

    # TODO: Implement function to do groupwise registration!!!!!

    # def store(self, directory):
    #     # TODO: Check that name is correct.
    #     pass

    @classmethod
    def from_directory(cls, directory):
        sub_dirs = sorted(os.listdir(directory), key=lambda x: int(x))
        reg_results = OrderedDict()
        for sub_dir in sub_dirs:
            reg_result = RegResult.from_directory(join(directory, sub_dir))
            reg_results[sub_dir] = reg_result
        return cls(reg_results)


@dataclass
class RegResult:
    path_to_small_affine: str
    path_to_big_affine: str
    path_to_small_warp: str
    path_to_big_warp: str
    path_to_small_inv_warp: str
    path_to_big_inv_warp: str
    width_downscaling_factor: float
    height_downscaling_factor: float
    path_to_small_fixed: str
    path_to_small_moving: str
    cmdl_log: Optional[List[subprocess.CompletedProcess]]
    reg_params: Optional[Any]
    path_to_small_ref_image: str
    sub_dir_key: int

    # def store(self, directory):
    #     # TODO: Do I even need that??? Would be more of a copy operation.
    #     pass

    @classmethod
    def from_directory(cls, directory):
        if not exists(directory):
            raise Exception(f'Could not load transformation. Directory {directory} not found.')
        with open(join(directory, 'reg_params.json')) as f:
            reg_params = json.load(f)
        path_to_big_warp = join(directory, 'metrics/full_resolution/big_warp.nii.gz')
        path_to_big_inv_warp = join(directory, 'metrics/full_resolution/big_inv_warp.nii.gz')
        path_to_big_affine = join(directory, 'metrics/full_resolution/Affine.mat')
        path_to_small_affine = join(directory, 'metrics/small_resolution/small_affine.mat')
        path_to_small_warp = join(directory, 'metrics/small_resolution/small_warp.nii.gz')
        path_to_small_inv_warp = join(directory, 'metrics/small_resolution/small_inv_warp.nii.gz')
        path_to_small_ref_image = join(directory, 'small_ref_image.nii.gz')
        width_downscaling_factor = reg_params['width_downscaling_factor']
        height_downscaling_factor = reg_params['height_downscaling_factor']
        sub_dir_key = int(os.path.split(os.path.normpath(directory))[1])
        return cls(
            path_to_small_affine=path_to_small_affine,
            path_to_big_affine=path_to_big_affine,
            path_to_small_warp=path_to_small_warp,
            path_to_big_warp=path_to_big_warp,
            path_to_small_inv_warp=path_to_small_inv_warp,
            path_to_big_inv_warp=path_to_big_inv_warp,
            width_downscaling_factor=width_downscaling_factor,
            height_downscaling_factor=height_downscaling_factor,
            path_to_small_fixed='',
            path_to_small_moving='',
            cmdl_log=None,
            reg_params=reg_params,
            path_to_small_ref_image=path_to_small_ref_image,
            sub_dir_key=sub_dir_key
        )


@dataclass
class MultiPointCloudTransformationResult:
    point_cloud_transformation_results: List[Any]#List['PointcloudTransformationResult']
    final_transform: Any

@dataclass
class PointcloudTransformationResult:
    pointcloud: Optional[Any]
    pointcloud_path: str
    cmdl_log: Optional[List[subprocess.CompletedProcess]]
    transform_params: Optional[Any]

@dataclass
class MultiGeojsonTransformationResult:
    multi_pointcloud_transformation_result: MultiPointCloudTransformationResult
    final_transform: Any

@dataclass
class MultiImageTransformationResult:
    image_transform_results: List[Any]#List['ImageTransformationResult']
    final_transform: Any

@dataclass
class ImageTransformationResult:
    registered_image: Optional[numpy.array]
    registered_image_path: str
    cmdl_log: Optional[List[subprocess.CompletedProcess]]
    transform_params: Optional[Any]

@dataclass
class GeojsonTransformationResult:
    transformed_geojson: Any
    pointcloud_transformation_result: PointcloudTransformationResult


@dataclass
class PreprocessedData:
    image_path: str
    height: int
    width: int
    height_padded: int
    width_padded: int
    height_original: int
    width_original: int


def preprocess_image(image: numpy.array,
                     kernel: int,
                     resolution: Tuple[int, int],
                     smoothing: int,
                     tmp_dir: str):
    small_image = resample_image_with_gaussian(image, resolution, smoothing)
    height_image = image.shape[1]
    width_image = image.shape[0]

    # 2. Preprocessing Step 2: Add padding for out of bounds. Estimate padded background by taking pixel information from background.

    height_small_image = small_image.shape[1]
    width_small_image = small_image.shape[0]

    img_padded = pad_image(small_image, kernel * 4)
    img_padded_sitk = sitk.GetImageFromArray(img_padded, isVector=True)
    img_padded_sitk.SetOrigin((4 * kernel, 4 * kernel))
    direction = tuple(map(lambda x: x * -1, img_padded_sitk.GetDirection()))
    img_padded_sitk.SetDirection(direction)

    width_image_padded = img_padded_sitk.GetWidth()
    height_image_padded = img_padded_sitk.GetHeight()
    path_small_image = join(tmp_dir, 'new_small_image.nii.gz')
    sitk.WriteImage(img_padded_sitk, path_small_image)

    preprocessed_data = PreprocessedData(
        image_path=path_small_image,
        height=height_small_image,
        width=width_small_image,
        height_padded=height_image_padded,
        width_padded=width_image_padded,
        height_original=height_image,
        width_original=width_image
    )
    return preprocessed_data

def clean_if_exists(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        
def derive_subdir(directory, limit=1000):
    for subdir_num in range(limit):
        subdir = f'{directory}/{subdir_num}'
        if not exists(subdir):
            return subdir, subdir_num
    # TODO: Do better error handling here, but who has 1000 sections to register, really?!
    return subdir, subdir_num

# TODO: There is probably a better way to ensure the dtype of the image.
def correct_img_dtype(img):
    if np.issubdtype(img.dtype, np.floating):
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)
    return img

@dataclass
class GreedyFHist:

    name: str = 'GreedyFHist'
    path_to_greedy: str = ''
    path_temp: str = 'tmp'
    path_output: str = 'out'
    segmentation_function: Optional[Callable] = None

    def register(self,
                 moving_img: numpy.array,
                 fixed_img: numpy.array,
                 moving_img_mask: Optional[numpy.array] = None,
                 fixed_img_mask: Optional[numpy.array] = None,
                 options: Optional[Options] = None) -> Any:
        """Performs registration from moving_img to fixed_img. Optional tissue masks can be provided.
        Options are supplied via the options arguments.

        Args:
            moving_img (numpy.array): 
            fixed_img (numpy.array): 
            moving_img_mask (Optional[numpy.array], optional): Optional moving mask. Is otherwise derived automatically. Defaults to None.
            fixed_img_mask (Optional[numpy.array], optional): Optional fixed mask. Is otherwise dervied automatically. Defaults to None.
            options (Optional[Options], optional): Can be supplied. Otherwise default arguments are used. Defaults to None.

        Returns:
            Any: _description_
        """
        reg_result = self.register_(moving_img,
                                    fixed_img,
                                    moving_img_mask,
                                    fixed_img_mask,
                                    options)
        reg_results = OrderedDict()
        reg_results[0] = reg_result
        multi_reg_result = MultiRegResult(reg_results=reg_results)
        return multi_reg_result
    
    def register_(self,
                 moving_img: numpy.array,
                 fixed_img: numpy.array,
                 moving_img_mask: Optional[numpy.array] = None,
                 fixed_img_mask: Optional[numpy.array] = None,
                 options: Optional[Options] = None,                  
                 **kwargs: Dict) -> 'RegResult':
        if options is None:
            options = Options()
        # Step 1: Set up all the parameters and filenames as necessary.
        path_output = options.output_directory
        
        # path_output = args.get('output_dir', 'out')
        path_output = join(path_output, 'registrations')
        path_output, subdir_num = derive_subdir(path_output)
        create_if_not_exists(path_output)
        self.path_output = path_output
        path_temp = options.temporary_directory
        # path_temp = args.get('tmp_dir', join(path_output, 'tmp'))
        clean_if_exists(path_temp)
        path_temp, _ = derive_subdir(path_temp)
        create_if_not_exists(path_temp)
        self.path_temp = path_temp
        affine_use_denoising = options.affine_do_denoising
        deformable_use_denoising = options.deformable_do_denoising
        # TODO: Implement autodownsampling if not set to get images to not bigger than 2000px
        pre_downsampling_factor = options.pre_downsampling_factor
        original_moving_image_size = moving_img.shape
        original_fixed_image_size = fixed_img.shape

        reg_params = {'s1': options.greedy_opts.s1,
                      's2': options.greedy_opts.s2,
                      'iteration_rigid': options.greedy_opts.iteration_rigid,
                      'resolution': options.resolution,
                      'affine_use_denoising': options.affine_do_denoising,
                      'deformable_use_denoising': options.deformable_do_denoising,
                      'options': options.to_dict(),
                      'pre_downsampling_factor': options.pre_downsampling_factor,
                      'original_moving_image_size': original_moving_image_size,
                      'original_fixed_image_size': original_fixed_image_size
                      }

        cmdln_returns = []
        # try:
        # Convert to correct format, if necessary
        # print(moving_img.dtype)
        moving_img = correct_img_dtype(moving_img)
        # print(moving_img.dtype)
        fixed_img = correct_img_dtype(fixed_img)
        moving_img = resample_by_factor(moving_img, pre_downsampling_factor)
        fixed_img = resample_by_factor(fixed_img, pre_downsampling_factor)
        if moving_img_mask is None:
            moving_img_mask = self.segmentation_function(moving_img)
        if fixed_img_mask is None:
            fixed_img_mask = self.segmentation_function(fixed_img)
        # Do padding
        # Missing step: Cropping around Tissue edges before. Remember to add the removed area to warped image later on.
        cropped_moving_mask, crop_params_mov = cropping(moving_img_mask)
        cropped_fixed_mask, crop_params_fix = cropping(fixed_img_mask)
        reg_params['cropping_params_mov'] = crop_params_mov
        reg_params['cropping_params_fix'] = crop_params_fix
        reg_params['original_shape_fixed_image'] = fixed_img.shape
        cropped_moving_img = moving_img[crop_params_mov[0]:crop_params_mov[1], crop_params_mov[2]:crop_params_mov[3]]
        cropped_fixed_img = fixed_img[crop_params_fix[0]:crop_params_fix[1], crop_params_fix[2]:crop_params_fix[3]]
        # print(cropped_fixed_img.shape, cropped_fixed_mask.shape, fixed_img.shape, fixed_img_mask.shape)        
        moving_pad, fixed_pad = get_symmetric_padding(cropped_moving_img, cropped_fixed_img)
        moving_img = pad_asym(cropped_moving_img, moving_pad)
        fixed_img = pad_asym(cropped_fixed_img, fixed_pad)
        moving_img_mask = pad_asym(cropped_moving_mask, moving_pad)
        fixed_img_mask = pad_asym(cropped_fixed_mask, fixed_pad)
        reg_params['moving_padding'] = moving_pad
        reg_params['fixed_padding'] = fixed_pad
        # print(cropped_fixed_img.shape, cropped_fixed_mask.shape, fixed_img.shape, fixed_img_mask.shape)
        moving_img = apply_mask(moving_img, moving_img_mask)
        fixed_img = apply_mask(fixed_img, fixed_img_mask)

        resample = options.resolution[0] / moving_img.shape[0] * 100
        smoothing = max(int(100 / (2 * resample)), 1) + 1
        reg_params['resample'] = resample
        reg_params['smoothing'] = smoothing

        requires_denoising = options.affine_do_denoising or options.deformable_do_denoising
        requires_standard_preprocessing = (not options.affine_do_denoising) or (not options.deformable_do_denoising)
        if requires_denoising:
            moving_img_denoised = denoise_image(moving_img, 
                                                sp=options.moving_sr, 
                                                sr=options.moving_sp)

            fixed_img_denoised = denoise_image(fixed_img, 
                                               sp=options.fixed_sp, 
                                               sr=options.fixed_sr)

            moving_denoised_tmp_dir = join(path_temp, 'moving_denoised')
            create_if_not_exists(moving_denoised_tmp_dir)
            moving_denoised_preprocessed = preprocess_image(moving_img_denoised, 
                                                            options.greedy_opts.kernel_size, 
                                                            options.resolution,
                                                            smoothing, 
                                                            moving_denoised_tmp_dir)
            fixed_denoised_tmp_dir = join(path_temp, 'fixed_denoised')
            create_if_not_exists(fixed_denoised_tmp_dir)
            fixed_denoised_preprocessed = preprocess_image(fixed_img_denoised, 
                                                           options.greedy_opts.kernel_size, 
                                                           options.resolution,
                                                           smoothing,
                                                           fixed_denoised_tmp_dir)
            height = fixed_denoised_preprocessed.height
        # 1. Set up directories and filenames
        if requires_standard_preprocessing:
            # Metrics
            moving_tmp_dir = join(path_temp, 'moving')
            create_if_not_exists(moving_tmp_dir)
            moving_img_preprocessed = preprocess_image(moving_img, 
                                                       options.greedy_opts.kernel_size, 
                                                       options.resolution, 
                                                       smoothing,
                                                       moving_tmp_dir)
            fixed_tmp_dir = join(path_temp, 'fixed')
            create_if_not_exists(fixed_tmp_dir)
            fixed_img_preprocessed = preprocess_image(fixed_img, 
                                                      options.greedy_opts.kernel_size, 
                                                      options.resolution, 
                                                      smoothing,
                                                      fixed_tmp_dir)
            height = fixed_img_preprocessed.height

        path_metrics = os.path.join(path_output, 'metrics')
        path_metrics_small_resolution = os.path.join(path_metrics, 'small_resolution')
        path_metrics_full_resolution = os.path.join(path_metrics, 'full_resolution')

        create_if_not_exists(path_output)
        # create_if_not_exists(path_temp)
        create_if_not_exists(path_metrics)
        create_if_not_exists(path_metrics_small_resolution)
        create_if_not_exists(path_metrics_full_resolution)

        # 3. Registration

        # Affine registration
        path_small_affine = os.path.join(path_metrics_small_resolution, 'small_affine.mat')
        offset = int((height + (options.greedy_opts.kernel_size * 4)) / 10)
        # iteration_vec = [100, 50, 10]
        reg_params['offset'] = offset
        reg_params['affine_iteration_vec'] = options.greedy_opts.affine_pyramid_iterations
        reg_params['deformable_iteration_vec'] = options.greedy_opts.deformable_pyramid_iterations

        ia_init = ''
        if options.greedy_opts.ia == 'ia-com-init' and fixed_img_mask is not None and moving_img_mask is not None:
            # Use Segmentation masks to compute center of mass initialization
            # Check that masks are np.arrays
            init_mat_path = os.path.join(path_temp, 'Affine_init.mat')
            init_mat = com_affine_matrix(fixed_img_mask, moving_img_mask)
            write_mat_to_file(init_mat, init_mat_path)
            ia_init = ['-ia', f'{init_mat_path}']
        elif options.greedy_opts.ia == 'ia-image-centers':
            ia_init = ['-ia-image-centers', '']
        else:
            print(f'Unknown ia option: {options.greedy_opts.ia}.')

        if affine_use_denoising:
            current_fixed_preprocessed = fixed_denoised_preprocessed
            current_moving_preprocessed = moving_denoised_preprocessed
            fixed_img_path = fixed_denoised_preprocessed.image_path
            moving_img_path = moving_denoised_preprocessed.image_path
        else:
            current_fixed_preprocessed = fixed_img_preprocessed
            current_moving_preprocessed = moving_img_preprocessed
            fixed_img_path = fixed_img_preprocessed.image_path
            moving_img_path = moving_img_preprocessed.image_path

        aff_ret = affine_registration(self.path_to_greedy,
                                      fixed_img_path,
                                      moving_img_path,
                                      path_small_affine,
                                      offset,
                                      ia_init,
                                      options.greedy_opts
                                      )
        # TODO: Add error handling for affine and deformable registration.

        cmdln_returns.append(aff_ret)

        # Diffeomorphic
        if affine_use_denoising and not deformable_use_denoising:
            # Map paths back
            fixed_img_path = fixed_img_preprocessed.image_path
            moving_img_path = moving_img_preprocessed.image_path
            current_fixed_preprocessed = fixed_img_preprocessed
            current_moving_preprocessed = moving_img_preprocessed

        path_small_warp = os.path.join(path_metrics_small_resolution, 'small_warp.nii.gz')
        path_small_warp_inv = os.path.join(path_metrics_small_resolution, 'small_inv_warp.nii.gz')

        deformable_reg_ret = deformable_registration(self.path_to_greedy,
                                                     fixed_img_path,
                                                     moving_img_path,
                                                     options.greedy_opts,
                                                     output_warp=path_small_warp,
                                                     output_inv_warp=path_small_warp_inv,
                                                     affine_pre_transform=path_small_affine)
        cmdln_returns.append(deformable_reg_ret)

        factor = 100 / resample
        reg_params['factor'] = factor

        path_affine = os.path.join(path_metrics_full_resolution, 'Affine.mat')
        rescale_affine(path_small_affine, path_affine, factor)

        reg_params['WIDTH_small_fixed'] = current_fixed_preprocessed.width
        reg_params['HEIGHT_small_fixed'] = current_fixed_preprocessed.height
        reg_params['WIDTH_fixed_image_padded'] = current_fixed_preprocessed.width_padded
        reg_params['HEIGHT_fixed_image_padded'] = current_fixed_preprocessed.height_padded
        reg_params['WIDTH_fixed_image'] = current_fixed_preprocessed.width_original
        reg_params['HEIGHT_fixed_image'] = current_fixed_preprocessed.height_original

        path_big_warp = os.path.join(path_metrics_full_resolution, 'big_warp.nii.gz')
        rescale_warp(path_small_warp,
                     path_big_warp,
                     (current_fixed_preprocessed.width, current_fixed_preprocessed.height),
                     (current_fixed_preprocessed.width_original,
                      current_fixed_preprocessed.height_original),
                     factor)
        reg_params['WIDTH_small_moving'] = current_moving_preprocessed.width
        reg_params['HEIGHT_small_moving'] = current_moving_preprocessed.height
        reg_params['WIDTH_moving_image_padded'] = current_moving_preprocessed.width_padded
        reg_params['HEIGHT_moving_image_padded'] = current_moving_preprocessed.height_padded
        reg_params['WIDTH_moving_image'] = current_moving_preprocessed.width_original
        reg_params['HEIGHT_moving_image'] = current_moving_preprocessed.height_original

        path_big_warp_inv = os.path.join(path_metrics_full_resolution, 'big_warp_inv.nii.gz')
        rescale_warp(
            path_small_warp_inv,
            path_big_warp_inv,
            (current_moving_preprocessed.width, current_moving_preprocessed.height),
            (current_moving_preprocessed.width_original,
             current_moving_preprocessed.height_original),
            factor)


        
        # Check those 2
        width_downscale = current_fixed_preprocessed.width / current_fixed_preprocessed.width_original
        height_downscale = current_fixed_preprocessed.height / current_fixed_preprocessed.height_original

        reg_params['width_downscaling_factor'] = width_downscale
        reg_params['height_downscaling_factor'] = height_downscale

        # Write small ref image to file for warping of coordinates
        small_fixed = sitk.ReadImage(current_fixed_preprocessed.image_path)
        empty_fixed_img = small_fixed[:,:]
        empty_fixed_img[:,:] = 0
        path_to_small_ref_image = join(path_output, 'small_ref_image.nii.gz')
        sitk.WriteImage(empty_fixed_img, path_to_small_ref_image)

        reg_param_outpath = os.path.join(path_output, 'reg_params.json')
        with open(reg_param_outpath, 'w') as f:
            json.dump(reg_params, f)

        if options.store_cmdline_returns:
            cmd_output = os.path.join(path_output, 'cmdl_returns.txt')
            with open(cmd_output, 'w') as f:
                for ret in cmdln_returns:
                    f.write(f'{ret}\n')

        reg_result = RegResult(
            path_to_small_affine=path_small_affine,
            path_to_big_affine=path_affine,
            path_to_small_warp=path_small_warp,
            path_to_big_warp=path_big_warp,
            path_to_small_inv_warp=path_small_warp_inv,
            path_to_big_inv_warp=path_big_warp_inv,
            width_downscaling_factor=width_downscale,
            height_downscaling_factor=height_downscale,
            path_to_small_moving=current_moving_preprocessed.image_path,
            path_to_small_fixed=current_fixed_preprocessed.image_path,
            cmdl_log=cmdln_returns,
            reg_params=reg_params,
            path_to_small_ref_image=path_to_small_ref_image,
            sub_dir_key=subdir_num
        )
        
        if options.remove_temporary_directory:
            self.__cleanup_temporary_directory(path_temp)        
        
        return reg_result

    def register_multi_image(self,
                             image_mask_list: List[Tuple[numpy.array, Optional[numpy.array]]],
                             options: Optional[Options] = None,
                             **kwargs: Dict):
        moving_image, moving_mask = image_mask_list[0]
        reg_results = OrderedDict()
        for (fixed_image, fixed_mask) in image_mask_list[1:]:
            reg_result = self.register_(moving_image,
                                        fixed_image,
                                        moving_mask,
                                        fixed_mask,
                                        options,
                                        **kwargs)
            image_transform_result = self.transform_image_(moving_image, reg_result, 'LINEAR')
            moving_image = image_transform_result.registered_image
            if moving_mask is not None:
                mask_transform_result = self.transform_image_(moving_mask, reg_result, 'NN')
                moving_mask = mask_transform_result.registered_image
            reg_results[reg_result.sub_dir_key] = reg_result
        multi_reg_result = MultiRegResult(reg_results=reg_results)
        return multi_reg_result

    def transform_image(self,
                        image: numpy.array,
                        transformation: MultiRegResult,
                        interpolation_mode: str,
                        **kwargs: Dict) -> 'MultiImageTransformationResult':
        tmp_dir = kwargs.get('tmp_dir', 'tmp')
        transformed_image = image.copy()
        transformation_results = []
        for key in transformation.reg_results:
            sub_transform = transformation.reg_results[key]
            sub_args = kwargs.copy()
            sub_args['tmp_dir'] = f'{tmp_dir}/{key}'
            transformation_result = self.transform_image_(transformed_image, sub_transform, interpolation_mode, **sub_args)
            transformed_image = transformation_result.registered_image
            transformation_results.append(transformation_result)
        multi_image_transform_result = MultiImageTransformationResult(transformation_results, transformation_results[-1])
        return multi_image_transform_result

    def transform_image_new_(self,
                             image: numpy.array,
                             transformation: RegResult,
                             interpolation_mode: str,
                             **kwargs: Dict) -> Any:
        """
        New implementation for warping data. Should limit dependence on Greedy and would work natively in Python.
        Probably means that we can throw away a lot of other parameters and Pointset transformation doesnt need the whole downscaling bit.
        For groupwise, composite ALL transformations to limit interpolation artifacts.
        Also for groupwise: Reimplement, so that we transform from end to beginning.
        
        Method: Load all transformation matrices, composite them using SimpleITK, warp. 
        """
        pass


    # Previously named warp_image
    def transform_image_(self,
                   image: numpy.array,
                   transformation: RegResult,
                   interpolation_mode: str,
                   **kwargs: Dict) -> Any:
        # TODO: Is this command just copying images into a new folder?
        tmp_dir = kwargs.get('tmp_dir', 'tmp')
        create_if_not_exists(tmp_dir)
        remove_temp_directory = kwargs.get('remove_temp_directory', True)
        output_image_path = join(tmp_dir, 'registered_image.nii.gz')
        image_dtype = image.dtype
        out_image_pixel_type = sitk.sitkUInt64 if image.dtype.name == 'uint64' else -1
        cmdl_returns = []
        warp_params = {}
        warp_params['out_image_pixel_type'] = out_image_pixel_type
        # Setup reference image ---- Not using a reference image anymore. Instead, just use the moving image. This only works if we assume that reference space is
        # the same as the moving image space, i.e. the same resolution.

        moving_fname = 'transform_moving.nii.gz'
        # moving_old_fname = 'transform_old_moving.png'
        moving_old_fname = 'transform_old_moving.nii.gz'
        path_old_source = os.path.join(tmp_dir, moving_old_fname)
        path_new_source = os.path.join(tmp_dir, moving_fname)

        # 1. Apply cropping from moving params
        original_image_shape = image.shape
        image = resample_by_factor(image, transformation.reg_params['pre_downsampling_factor'])
        cropping_mov = transformation.reg_params['cropping_params_mov']
        image = image[cropping_mov[0]:cropping_mov[1], cropping_mov[2]:cropping_mov[3]]
        # Add padding to image
        # print(f"""Image size: {image.shape} and padding: {transformation.reg_params['moving_padding']}.""")
        image = pad_asym(image, transformation.reg_params['moving_padding'])
        # print(f"""Image after padding: {image.shape}""")
        # image = pad_image(image, transformation.reg_params['moving_img_padding'])
        sitk_image = sitk.GetImageFromArray(image, True)
        direction = tuple(map(lambda x: x * -1, sitk_image.GetDirection()))
        sitk_image.SetDirection(direction)
        sitk.WriteImage(sitk_image, path_new_source)
        # cmd = f"""{self.path_to_c2d} -mcs '{path_old_source}' -foreach -orient LP -spacing 1x1mm -origin 0x0mm -endfor -omc '{path_new_source}'"""
        # moving_pre_ret = call_command(cmd)
        # cmdl_returns.append(moving_pre_ret)

        transforms = [transformation.path_to_big_warp, transformation.path_to_big_affine]
        warp_params['transforms'] = transforms
        # interpolation_mode = args.get('interpolation_mode', 'LINEAR')
        rb = kwargs.get('rb', 0)
        warp_params['interpolation_mode'] = interpolation_mode
        big_reslice_args = {}
        big_reslice_args['-r'] = transforms
        big_reslice_args['-d'] = '2'
        big_reslice_args['-rf'] = path_new_source
        big_reslice_args['-rm'] = [path_new_source, output_image_path]
        big_reslice_args['-ri'] = [interpolation_mode]
        big_reslice_args['-rb'] = f'{rb}'

        big_reslice_cmd = build_cmd_string(self.path_to_greedy, big_reslice_args)
        big_reslice_ret = call_command(big_reslice_cmd)
        registered_image = sitk.GetArrayFromImage(sitk.ReadImage(output_image_path, out_image_pixel_type))
        # We get images back that are floats in the 0-255 range.
        # registered_image = registered_image.astype(int)
        registered_image = registered_image.astype(image_dtype)
        # registered_image = registered_image.astype(np.int32)
        # If the image has 3 colour channels convert back to ints.
        if len(registered_image.shape) == 2:
            registered_image = registered_image.astype(np.int32)
        # print(f"""Shape of image after registration: {registered_image.shape} and padding: {transformation.reg_params['fixed_padding']}""")
        registered_image = remove_padding(registered_image, transformation.reg_params['fixed_padding'])
        # Add cropping from fixed image
        cropping_fixed = transformation.reg_params['cropping_params_fix']
        registered_image = add_cropped_region(registered_image, transformation.reg_params['original_shape_fixed_image'], cropping_fixed)
        # Resize to original size
        target_size = transformation.reg_params['original_fixed_image_size']
        registered_image = resize_image(registered_image, target_size, interpolation_mode)

        
        # print(f'Shape after removing padding: {registered_image.shape}')
        cmdl_returns.append(big_reslice_ret)
        if remove_temp_directory:
            self.__cleanup_temporary_directory(tmp_dir)
        transform_ret = ImageTransformationResult(
            registered_image=registered_image,
            registered_image_path=output_image_path,
            cmdl_log=cmdl_returns,
            transform_params=warp_params
        )
        return transform_ret

    def transform_pointset(self,
                           pointset: pandas.DataFrame,
                           transformation: MultiRegResult,
                           **kwargs: Dict) -> 'MultiPointCloudTransformationResult':
        """Transforms pointset. Can be applied to chained registrations.

        Args:
            pointset (pandas.DataFrame):
            transformation (MultiRegResult): _description_
            args (_type_, optional): _description_. Defaults to None.

        Returns:
            Any: _description_
        """
        tmp_dir = kwargs.get('tmp_dir', 'tmp')
        transformed_coordinates = pointset.copy()
        transformation_results = []
        for key in transformation.reg_results:
            sub_transform = transformation.reg_results[key]
            sub_args = kwargs.copy()
            sub_args['tmp_dir'] = f'{tmp_dir}/{key}'
            transformation_result = self.transform_pointset_(transformed_coordinates, sub_transform, **sub_args)
            transformed_coordinates = transformation_result.pointcloud
            transformation_results.append(transformation_result)
        multi_image_transform_result = MultiPointCloudTransformationResult(transformation_results, transformation_results[-1])
        return multi_image_transform_result
        

    def transform_pointset_(self,
                         pointset: pandas.DataFrame,
                         transformation: RegResult,
                         **kwargs) -> Any:
        """Transform pointset. 

        Args:
            pointset (pandas.DataFrame): Supplied as a dataframe. x and y columns are used for transforming data.
            transformation (RegResult): 
            args (_type_, optional): Any additional arguments for transforming. Defaults to None.

        Returns:
            Any: _description_
        """
        tmp_dir = kwargs.get('tmp_dir', 'tmp')
        output_path = kwargs.get('output_path', join(tmp_dir, 'registered_pc.csv'))
        create_if_not_exists(tmp_dir)
        remove_temp_directory = kwargs.get('remove_temp_directory', True)
        path_landmarks = os.path.join(tmp_dir, 'coordinates.csv')
        pre_downsampling_factor = transformation.reg_params['pre_downsampling_factor']
        pointset = scale_table(pointset, pre_downsampling_factor)
        pointset.to_csv(path_landmarks)
        width_downscale = transformation.width_downscaling_factor
        height_downscale = transformation.height_downscaling_factor
        cropping_params_mov = transformation.reg_params['cropping_params_mov']
        cropping_params_fix = transformation.reg_params['cropping_params_fix']
        paths_to_inv_transforms = [f'{transformation.path_to_small_affine},-1', transformation.path_to_small_inv_warp]
        transform_res = process_and_warp_coordinates(self.path_to_greedy,
                                                     path_landmarks,
                                                     output_path,
                                                     tmp_dir,
                                                     paths_to_inv_transforms,
                                                     transformation.path_to_small_ref_image,
                                                     width_downscale,
                                                     height_downscale,
                                                     transformation.reg_params['moving_padding'],
                                                     transformation.reg_params['fixed_padding'],
                                                     cropping_params_mov,
                                                     cropping_params_fix)
        warped_coordinates = scale_table(transform_res.pointcloud, 1 / pre_downsampling_factor)
        transform_res.pointcloud = warped_coordinates
        if remove_temp_directory:
            self.__cleanup_temporary_directory(tmp_dir)
        return transform_res

    def transform_geojson(self,
                          geojson_data: geojson.GeoJSON,
                          transformation: MultiRegResult,
                          **kwargs) -> 'MultiGeojsonTransformationResult':
        """Transforms geojson data using the computed transformation.

        Args:
            geojson_data (geojson.Geojson): Geojson data to be transformed.
            transformation (MultiRegResult): Transformation computed by registration.
            args (_type_, optional): Any optional additional arguments. Defaults to None.

        Returns:
            MultiGeojsonTransformationResult: Warped geojson data. 
        """
        geojson_data = copy.deepcopy(geojson_data)
        geo_df = geojson_2_table(geojson_data)
        transform_result = self.transform_pointset(geo_df, transformation, **kwargs)
        warped_geo_coordinates_df = transform_result.final_transform
        warped_geo_df = geo_df.copy()
        warped_geo_df.x = warped_geo_coordinates_df.pointcloud.iloc[:, 0].to_numpy()
        warped_geo_df.y = warped_geo_coordinates_df.pointcloud.iloc[:, 1].to_numpy()
        warped_geojson = convert_table_2_geo_json(geojson_data, warped_geo_df)
        geojson_transform_result = GeojsonTransformationResult(warped_geojson, transform_result)
        geojson_multi_transform_result = MultiGeojsonTransformationResult(transform_result.point_cloud_transformation_results, geojson_transform_result)
        return geojson_multi_transform_result

    def __cleanup_temporary_directory(self, directory: str) -> None:
        """Removes the temporary directory.

        Args:
            directory (str):
        """
        shutil.rmtree(directory)

    @classmethod
    def load_from_config(cls, config: Dict[str, Any]) -> 'GreedyFHist':
        # Refers to greedy's directory. If not supplied, assumes that greedy is in PATH.
        path_to_greedy = config.get('path_to_greedy', '')
        path_to_greedy = join(path_to_greedy, 'greedy')
        seg_fun = load_yolo_segmentation()
        return cls(path_to_greedy=path_to_greedy, segmentation_function=seg_fun)
