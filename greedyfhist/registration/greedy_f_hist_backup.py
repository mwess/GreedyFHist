from collections import OrderedDict
from dataclasses import dataclass
import copy
import json
import os
from os.path import join, exists
import shutil
import subprocess
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy
import numpy as np
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


def get_default_args():
    return {
        'kernel': 10,
        'resolution': (1024, 1024),
        'use_segmentation_masks': True,
        'output_dir': 'save_directories/temp_nb/',
        'tmp_dir': 'save_directories/temp_nb/tmp',
        'cleanup_temporary_directories': False,
        'remove_temp_directory': False,
        'cost_fun': 'WNCC',
        'ia': 'ia-com-init',
        'affine_use_denoising': True,
        'deformable_use_denoising': True,
        'pre_downsampling_factor': 1
    }


def deformable_registration(path_to_greedy,
                            path_fixed_image,
                            path_moving_image,
                            cost_fun,
                            kernel,
                            iteration_vec,
                            s1,
                            s2,
                            output_warp=None,
                            output_inv_warp=None,
                            affine_pre_transform=None,
                            additional_args=None
                            ):
    cost_fun_params = cost_fun
    if cost_fun == 'ncc' or cost_fun == 'wncc':
        cost_fun_params += f' {kernel}x{kernel}'
    def_args = {}
    def_args['-it'] = affine_pre_transform
    def_args['-d'] = '2'
    # def_args['-m'] = f'NCC {kernel}x{kernel}'
    def_args['-m'] = cost_fun_params
    def_args['-i'] = [path_fixed_image, path_moving_image]
    def_args['-n'] = f'{iteration_vec[0]}x{iteration_vec[1]}x{iteration_vec[2]}'
    def_args['-threads'] = '32'
    def_args['-s'] = [f'{s1}vox', f'{s2}vox']
    def_args['-o'] = output_warp
    def_args['-oinv'] = output_inv_warp
    if '-sv' in additional_args:
        def_args['-sv'] = ''
    elif '-svlb' in additional_args:
        def_args['-svlb'] = ''

    def_cmd = build_cmd_string(path_to_greedy, def_args)
    def_ret = call_command(def_cmd)
    return def_ret


def affine_registration(path_to_greedy,
                        path_to_fixed_image,
                        path_to_moving_image,
                        path_output,
                        offset,
                        iteration_vec,
                        iteration_rigid,
                        cost_fun,
                        ia,
                        kernel=None
                        ):
    cost_fun_params = cost_fun
    if cost_fun == 'ncc' or cost_fun == 'wncc':
        cost_fun_params += f' {kernel}x{kernel}'
    aff_rgs = {}
    aff_rgs['-d'] = '2'
    aff_rgs['-i'] = [path_to_fixed_image, path_to_moving_image]
    aff_rgs['-o'] = path_output
    aff_rgs['-m'] = cost_fun_params
    aff_rgs['-n'] = f'{iteration_vec[0]}x{iteration_vec[1]}x{iteration_vec[2]}'
    aff_rgs['-threads'] = '32'
    aff_rgs['-dof'] = '12'
    aff_rgs['-search'] = f'{iteration_rigid} 180 {offset}'.split()  # Replaced 360 with any for rotation parameter
    aff_rgs['-gm-trim'] = f'{kernel}x{kernel}'
    aff_rgs['-a'] = ''  # Doesnt get param how to parse?
    aff_rgs[ia[0]] = ia[1]

    aff_cmd = build_cmd_string(path_to_greedy, aff_rgs)
    aff_ret = call_command(aff_cmd)
    return aff_ret


# Warp functions

def warp_image_data(fixed_image_path,
                    moving_image_path,
                    output_image_path,
                    transforms,
                    path_to_greedy,
                    path_to_c2d,
                    temp_dir,
                    interpolation_mode='LINEAR'):
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


def map_infs_back(warped_landmarks, source_landmarks):
    infs = (source_landmarks.x_small == np.inf) | (source_landmarks.y_small == np.inf)
    infs = infs.to_numpy()
    warped_landmarks.x.loc[infs] = np.inf
    warped_landmarks.y.loc[infs] = np.inf
    return warped_landmarks


def process_and_warp_coordinates(path_to_greedy,
                                 path_coordinates,
                                 output_path,
                                 path_temp,
                                 paths_to_inv_transforms,
                                 path_small_ref_img,
                                 width_downscale,
                                 height_downscale,
                                 moving_padding,
                                 fixed_padding,
                                 cropped_params_mov,
                                 cropped_params_fix,
                                 additional_args=None):
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
                 args: Optional[Dict[Any, Any]] = None) -> Any:
        reg_result = self.register_(moving_img,
                                    fixed_img,
                                    moving_img_mask,
                                    fixed_img_mask,
                                    args)
        reg_results = OrderedDict()
        reg_results[0] = reg_result
        multi_reg_result = MultiRegResult(reg_results=reg_results)
        return multi_reg_result

    def register_(self,
                 moving_img: numpy.array,
                 fixed_img: numpy.array,
                 moving_img_mask: Optional[numpy.array] = None,
                 fixed_img_mask: Optional[numpy.array] = None,
                 args: Optional[Dict[Any, Any]] = None) -> Any:
        if args is None:
            args = {}
        # Step 1: Set up all the parameters and filenames as necessary.
        path_output = args.get('output_dir', 'out')
        path_output = join(path_output, 'registrations')
        path_output, subdir_num = derive_subdir(path_output)
        create_if_not_exists(path_output)
        self.path_output = path_output
        path_temp = args.get('tmp_dir', join(path_output, 'tmp'))
        clean_if_exists(path_temp)
        path_temp, _ = derive_subdir(path_temp)
        create_if_not_exists(path_temp)
        self.path_temp = path_temp
        cleanup_temporary_directories = args.get('cleanup_temporary_directories', True)
        s1 = args.get('s1', 6.0)
        s2 = args.get('s2', 5.0)
        resolution = tuple(map(lambda x: int(x), args.get('resolution', (1024, 1024))))
        kernel = args.get('kernel', 10)
        iteration_rigid = args.get('iteration_rigid', 10000)
        affine_init = args.get('ia', 'ia-image-centers')
        cost_fun = args.get('cost_fun', 'ncc').lower()
        store_cmdl_returns = args.get('store_cmdl_returns', True)
        affine_use_denoising = args.get('affine_use_denoising', True)
        deformable_use_denoising = args.get('deformable_use_denoising', False)
        pre_downsampling_factor = args.get('pre_downsampling_factor', 1)
        original_moving_image_size = moving_img.shape
        original_fixed_image_size = fixed_img.shape

        reg_params = {'s1': s1,
                      's2': s2,
                      'iteration_rigid': iteration_rigid,
                      'resolution': resolution,
                      'affine_use_denoising': affine_use_denoising,
                      'deformable_use_denoising': deformable_use_denoising,
                      'args': args,
                      'pre_downsampling_factor': pre_downsampling_factor,
                      'original_moving_image_size': original_moving_image_size,
                      'original_fixed_image_size': original_fixed_image_size
                      }

        cmdln_returns = []
        # try:
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

        resample = resolution[0] / moving_img.shape[0] * 100
        smoothing = max(int(100 / (2 * resample)), 1)
        reg_params['resample'] = resample
        reg_params['smoothing'] = smoothing

        requires_denoising = affine_use_denoising or deformable_use_denoising
        requires_standard_preprocessing = (not affine_use_denoising) or (not deformable_use_denoising)
        if requires_denoising:
            mov_sr = args.get('mov_sr', 30)
            mov_sp = args.get('mov_sp', 20)
            moving_img_denoised = denoise_image(moving_img, sp=mov_sp, sr=mov_sr)

            fix_sr = args.get('fix_sr', 30)
            fix_sp = args.get('fix_sp', 20)
            fixed_img_denoised = denoise_image(fixed_img, sp=fix_sp, sr=fix_sr)

            moving_denoised_tmp_dir = join(path_temp, 'moving_denoised')
            create_if_not_exists(moving_denoised_tmp_dir)
            moving_denoised_preprocessed = preprocess_image(moving_img_denoised, kernel, resolution,
                                                            smoothing, moving_denoised_tmp_dir)
            fixed_denoised_tmp_dir = join(path_temp, 'fixed_denoised')
            create_if_not_exists(fixed_denoised_tmp_dir)
            fixed_denoised_preprocessed = preprocess_image(fixed_img_denoised, kernel, resolution,
                                                           smoothing,
                                                           fixed_denoised_tmp_dir)
            height = fixed_denoised_preprocessed.height
        # 1. Set up directories and filenames
        if requires_standard_preprocessing:
            # Metrics
            moving_tmp_dir = join(path_temp, 'moving')
            create_if_not_exists(moving_tmp_dir)
            moving_img_preprocessed = preprocess_image(moving_img, kernel, resolution, smoothing,
                                                       moving_tmp_dir)
            fixed_tmp_dir = join(path_temp, 'fixed')
            create_if_not_exists(fixed_tmp_dir)
            fixed_img_preprocessed = preprocess_image(fixed_img, kernel, resolution, smoothing,
                                                      fixed_tmp_dir)
            height = fixed_img_preprocessed.height

        path_metrics = os.path.join(path_output, 'metrics')
        path_metrics_small_resolution = os.path.join(path_metrics, 'small_resolution')
        path_metrics_full_resolution = os.path.join(path_metrics, 'full_resolution')

        create_if_not_exists(path_output)
        create_if_not_exists(path_temp)
        create_if_not_exists(path_metrics)
        create_if_not_exists(path_metrics_small_resolution)
        create_if_not_exists(path_metrics_full_resolution)

        # 3. Registration

        # Affine registration
        path_small_affine = os.path.join(path_metrics_small_resolution, 'small_affine.mat')
        offset = int((height + (kernel * 4)) / 10)
        iteration_vec = [100, 50, 10]
        reg_params['offset'] = offset
        reg_params['iteration_vec'] = iteration_vec

        ia_init = ''
        if affine_init == 'ia-com-init' and fixed_img_mask is not None and moving_img_mask is not None:
            # Use Segmentation masks to compute center of mass initialization
            # Check that masks are np.arrays
            init_mat_path = os.path.join(path_temp, 'Affine_init.mat')
            init_mat = com_affine_matrix(fixed_img_mask, moving_img_mask)
            write_mat_to_file(init_mat, init_mat_path)
            ia_init = ['-ia', f'{init_mat_path}']
        elif affine_init == 'ia-image-centers':
            ia_init = ['-ia-image-centers', '']
        else:
            print(f'Unknown ia option: {affine_init}.')

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
                                      iteration_vec,
                                      iteration_rigid,
                                      cost_fun,
                                      ia=ia_init,
                                      kernel=kernel)
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
                                                     cost_fun,
                                                     kernel,
                                                     iteration_vec,
                                                     s1,
                                                     s2,
                                                     output_warp=path_small_warp,
                                                     output_inv_warp=path_small_warp_inv,
                                                     affine_pre_transform=path_small_affine,
                                                     additional_args=args)
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

        if cleanup_temporary_directories:
            self.__cleanup_temporary_directory(path_temp)
        
        # Check those 2
        width_downscale = current_fixed_preprocessed.width / current_fixed_preprocessed.width_original
        height_downscale = current_fixed_preprocessed.height / current_fixed_preprocessed.height_original

        reg_params['width_downscaling_factor'] = width_downscale
        reg_params['height_downscaling_factor'] = height_downscale

        # Write small ref image to file for warping of coordinates
        small_moving = sitk.ReadImage(current_moving_preprocessed.image_path)
        empty_moving_img = small_moving[:,:]
        empty_moving_img[:,:] = 0
        path_to_small_ref_image = join(path_output, 'small_ref_image.nii.gz')
        sitk.WriteImage(empty_moving_img, path_to_small_ref_image)

        reg_param_outpath = os.path.join(path_output, 'reg_params.json')
        with open(reg_param_outpath, 'w') as f:
            json.dump(reg_params, f)

        if store_cmdl_returns:
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
        return reg_result

    def register_multi_image(self,
                             image_mask_list: List[Tuple[numpy.array, Optional[numpy.array]]],
                             args):
        moving_image, moving_mask = image_mask_list[0]
        reg_results = OrderedDict()
        for (fixed_image, fixed_mask) in image_mask_list:
            reg_result = self.register_(moving_image,
                                        fixed_image,
                                        moving_mask,
                                        fixed_mask,
                                        args)
            reg_results[reg_results.sub_dir_key] = reg_result
        multi_reg_result = MultiRegResult(reg_results=reg_results)
        return multi_reg_result

    def transform_image(self,
                        image: numpy.array,
                        transformation: MultiRegResult,
                        interpolation_mode: str,
                        args: Optional[Dict[Any, Any]] = None) -> Any:
        if args is None:
            args = {}
        tmp_dir = args.get('tmp_dir', 'tmp')
        transformed_image = image.copy()
        transformation_results = []
        for key in transformation.reg_results:
            sub_transform = transformation.reg_results[key]
            sub_args = args.copy()
            sub_args['tmp_dir'] = f'{tmp_dir}/{key}'
            transformation_result = self.transform_image_(transformed_image, sub_transform, interpolation_mode, sub_args)
            transformed_image = transformation_result.registered_image
            transformation_results.append(transformation_result)
        multi_image_transform_result = MultiImageTransformationResult(transformation_results, transformation_results[-1])
        return multi_image_transform_result

    # Previously named warp_image
    def transform_image_(self,
                   image: numpy.array,
                   transformation: RegResult,
                   interpolation_mode: str,
                   args: Optional[Dict[Any, Any]] = None) -> Any:
        # TODO: Is this command just copying images into a new folder?
        if args is None:
            args = {}
        tmp_dir = args.get('tmp_dir', 'tmp')
        create_if_not_exists(tmp_dir)
        remove_temp_directory = args.get('remove_temp_directory', True)
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
        rb = args.get('rb', 0)
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
                              coordinates,
                              transformation: MultiRegResult,
                              args=None) -> Any:
        if args is None:
            args = {}
        tmp_dir = args.get('tmp_dir', 'tmp')
        transformed_coordinates = coordinates.copy()
        transformation_results = []
        for key in transformation.reg_results:
            sub_transform = transformation.reg_results[key]
            sub_args = args.copy()
            sub_args['tmp_dir'] = f'{tmp_dir}/{key}'
            transformation_result = self.transform_coordinates_(transformed_coordinates, sub_transform, sub_args)
            transformed_coordinates = transformation_result.pointcloud
            transformation_results.append(transformation_result)
        multi_image_transform_result = MultiPointCloudTransformationResult(transformation_results, transformation_results[-1])
        return multi_image_transform_result
        

    def transform_coordinates_(self,
                         coordinates,
                         transformation: RegResult,
                         args=None) -> Any:
        if args is None:
            args = {}
        tmp_dir = args.get('tmp_dir', 'tmp')
        output_path = args.get('output_path', join(tmp_dir, 'registered_pc.csv'))
        create_if_not_exists(tmp_dir)
        remove_temp_directory = args.get('remove_temp_directory', True)
        path_landmarks = os.path.join(tmp_dir, 'coordinates.csv')
        pre_downsampling_factor = transformation.reg_params['pre_downsampling_factor']
        coordinates = scale_table(coordinates, pre_downsampling_factor)
        coordinates.to_csv(path_landmarks)
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
                          geojson_data,
                          transformation: MultiRegResult,
                          args=None) -> Any:
        args = args if args is not None else {}
        geojson_data = copy.deepcopy(geojson_data)
        geo_df = geojson_2_table(geojson_data)
        transform_result = self.transform_pointset(geo_df, transformation, args)
        warped_geo_coordinates_df = transform_result.final_transform
        warped_geo_df = geo_df.copy()
        warped_geo_df.x = warped_geo_coordinates_df.pointcloud.iloc[:, 0].to_numpy()
        warped_geo_df.y = warped_geo_coordinates_df.pointcloud.iloc[:, 1].to_numpy()
        warped_geojson = convert_table_2_geo_json(geojson_data, warped_geo_df)
        geojson_transform_result = GeojsonTransformationResult(warped_geojson, transform_result)
        geojson_multi_transform_result = MultiGeojsonTransformationResult(transform_result.point_cloud_transformation_results, geojson_transform_result)
        return geojson_multi_transform_result

    # def transform_geojson_(self,
    #                       geojson_data,
    #                       transformation: RegResult,
    #                       args=None) -> Any:
    #     args = args if args is not None else {}
    #     downscale_factor = args.get('downscale_factor', 1)
    #     geo_df = geojson_2_table(geojson_data)
    #     geo_df.rename(columns={0: 'x', 1: 'y'}, inplace=True)
    #     geo_df = scale_table(geo_df, downscale_factor)
    #     transform_result = self.transform_coordinates(geo_df, transformation, args)
    #     warped_geo_df = geo_df.copy()
    #     warped_geo_df.x = transform_result.pointcloud.iloc[:, 0].to_numpy()
    #     warped_geo_df.y = transform_result.pointcloud.iloc[:, 1].to_numpy()
    #     warped_geo_df = scale_table(geo_df, 1 / downscale_factor)
    #     warped_geojson = convert_table_2_geo_json(geojson_data, warped_geo_df)
    #     geojson_transform_result = GeojsonTransformationResult(warped_geojson, transform_result)
    #     return geojson_transform_result
        

    def __cleanup_temporary_directory(self, directory):
        shutil.rmtree(directory)

    @classmethod
    def load_from_config(cls, config: Dict[str, Any]) -> 'GreedyFHist':
        # Refers to greedy's directory. If not supplied, assumes that greedy is in PATH.
        path_to_greedy = config.get('path_to_greedy', '')
        path_to_greedy = join(path_to_greedy, 'greedy')
        seg_fun = load_yolo_segmentation()
        return cls(path_to_greedy=path_to_greedy, segmentation_function=seg_fun)
