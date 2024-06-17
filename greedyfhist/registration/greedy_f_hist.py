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
import SimpleITK
import SimpleITK as sitk

from greedyfhist.utils.io import create_if_not_exists, write_mat_to_file, clean_if_exists
from greedyfhist.utils.image import (
    rescale_warp,
    rescale_affine_2,
    denoise_image,
    apply_mask,
    com_affine_matrix,
    pad_image,
    resample_image_with_gaussian,
    pad_asym,
    get_symmetric_padding,
    cropping,
    resample_by_factor,
    resample_image_sitk
)
from greedyfhist.utils.utils import deformable_registration, affine_registration, composite_warps
from greedyfhist.utils.geojson_utils import geojson_2_table, convert_table_2_geo_json
from greedyfhist.segmentation.segmenation import load_yolo_segmentation
from greedyfhist.options import RegistrationOptions


# TODO: Write serialization
@dataclass
class GroupwiseRegResult:

    affine_transform: List['GFHTransform']
    deformable_transform: List['GFHTransform']

    def get_transforms(self,
                       source: int,
                       no_deformable: bool = False):
        # TODO: At the moment only one direction works.
        transforms = self.affine_transform[source:]
        if self.deformable_transform is not None and len(self.deformable_transform) > 0:
            transforms.append(self.deformable_transform[source])
        # Composite transforms
        composited_fixed_transform = compose_transforms([x.fixed_transform for x in transforms])
        composited_moving_transform = compose_transforms([x.moving_transform for x in transforms][::-1])
        reg_result = RegistrationResult(composited_fixed_transform, composited_moving_transform)
        return reg_result
    

@dataclass
class GFHTransform:
    
    size: Tuple[int, int]
    transform: SimpleITK.SimpleITK.Transform

    # TODO: Check that path is directory and change name since we are storing to a directory and not to one file.
    def to_file(self, path):
        create_if_not_exists(path)
        attributes = {
            'width': self.size[0],
            'height': self.size[1]
        }
        attributes_path = join(path, 'attributes.json')
        with open(attributes_path, 'w') as f:
            json.dump(attributes, f)
        transform_path = join(path, 'transform.txt')
        self.transform.FlattenTransform()
        sitk.WriteTransform(self.transform, transform_path)


    @staticmethod
    def load_transform(path):
        attributes_path = join(path, 'attributes.json')
        with open(attributes_path) as f:
            attributes = json.load(f)
        size = (attributes['width'], attributes['height'])
        transform_path = join(path, 'transform.txt')
        transform = sitk.ReadTransform(transform_path)
        return GFHTransform(size, transform)


@dataclass
class RegistrationResult:
    
    fixed_transform: 'GFHTransform'
    moving_transform: 'GFHTransform'
    cmdln_returns: Optional[List[Any]] = None
    
    # TODO: Can I add cmdln_returns somehow
    def to_file(self, path):
        create_if_not_exists(path)
        fixed_transform_path = join(path, 'fixed_transform')
        self.fixed_transform.to_file(fixed_transform_path)
        moving_transform_path = join(path, 'moving_transform')
        self.moving_transform.to_file(moving_transform_path)

    @staticmethod
    def load(path):
        fixed_transform_path = join(path, 'fixed_transform')
        fixed_transform = GFHTransform.load_transform(fixed_transform_path)
        moving_transform_path = join(path, 'moving_transform')
        moving_transform = GFHTransform.load_transform(moving_transform_path)
        return RegistrationResult(fixed_transform, moving_transform)


# TODO: Rename that class (or remove it)
@dataclass
class RegResult:
    path_to_small_fixed: str
    path_to_small_moving: str
    path_to_small_composite: str
    path_to_big_composite: str
    path_to_small_inv_composite: str
    path_to_big_inv_composite: str
    cmdl_log: Optional[List[subprocess.CompletedProcess]]
    reg_params: Optional[Any]
    path_to_small_ref_image: str
    sub_dir_key: int
    displacement_field: SimpleITK.SimpleITK.Image
    inv_displacement_field: SimpleITK.SimpleITK.Image

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
class PreprocessedData:
    image_path: str
    height: int
    width: int
    height_padded: int
    width_padded: int
    height_original: int
    width_original: int


# TODO: Write a sub function that just works with simpleitk transforms.
def composite_sitk_transforms(transforms: List[SimpleITK.SimpleITK.Transform]) -> SimpleITK.SimpleITK.Transform:
    composited_transform = sitk.CompositeTransform(2)
    for transform in transforms:
        composited_transform.AddTransform(transform)
    return composited_transform
    

def compose_transforms(gfh_transforms: List[Any]) -> 'GFHTransform':
    composited_transform = composite_sitk_transforms([x.transform for x in gfh_transforms])
    gfh_comp_trans = GFHTransform(gfh_transforms[-1].size, composited_transform)
    return gfh_comp_trans

def compose_reg_transforms(transform: SimpleITK.SimpleITK.Transform, 
                           transformation: RegResult) -> SimpleITK.SimpleITK.Transform:
    # TODO: Rewrite this function such that the transformation to put in the middle is passed an as argument.
    

    # TODO: Implement transformation from fixed to moving space in case we want to go in a different direction?
    moving_padding = transformation.reg_params['moving_padding']
    moving_cropping = transformation.reg_params['cropping_params_mov']
    fixed_padding = transformation.reg_params['fixed_padding']
    fixed_cropping = transformation.reg_params['cropping_params_fix']
    fixed_image_shape = transformation.reg_params['original_fixed_image_size']

    all_transforms = sitk.CompositeTransform(2)
    # Try adding downsampling factor
    ds_factor = transformation.reg_params['pre_downsampling_factor']
    pre_downscale_transform = sitk.ScaleTransform(2, (1/ds_factor, 1/ds_factor))
    post_upscale_transform = sitk.ScaleTransform(2, (ds_factor, ds_factor))
    
    aff_trans1 = sitk.TranslationTransform(2)
    offset_x = -moving_padding[0]
    offset_y = -moving_padding[2]
    aff_trans1.SetOffset((offset_x, offset_y))
    
    aff_trans2 = sitk.TranslationTransform(2)
    offset_x = moving_cropping[2]
    offset_y = moving_cropping[0]
    aff_trans2.SetOffset((offset_x, offset_y))

    aff_trans3 = sitk.TranslationTransform(2)
    aff_trans3.SetOffset((-fixed_cropping[2], -fixed_cropping[0]))
    
    aff_trans4 = sitk.TranslationTransform(2)
    aff_trans4.SetOffset((fixed_padding[0], fixed_padding[2]))

    all_transforms.AddTransform(pre_downscale_transform)
    all_transforms.AddTransform(aff_trans1)
    all_transforms.AddTransform(aff_trans2)
    all_transforms.AddTransform(transform)
    all_transforms.AddTransform(aff_trans3)
    all_transforms.AddTransform(aff_trans4)
    all_transforms.AddTransform(post_upscale_transform)
    return all_transforms


def compose_inv_reg_transforms(transform: SimpleITK.SimpleITK.Transform, 
                               transformation: RegResult) -> SimpleITK.SimpleITK.Transform:

    moving_padding = transformation.reg_params['moving_padding']
    moving_cropping = transformation.reg_params['cropping_params_mov']
    fixed_padding = transformation.reg_params['fixed_padding']
    fixed_cropping = transformation.reg_params['cropping_params_fix']
    moving_image_shape = transformation.reg_params['original_moving_image_size']
    
    all_transforms = sitk.CompositeTransform(2)

    ds_factor = transformation.reg_params['pre_downsampling_factor']
    pre_downscale_transform = sitk.ScaleTransform(2, (1/ds_factor, 1/ds_factor))
    post_upscale_transform = sitk.ScaleTransform(2, (ds_factor, ds_factor))

    aff_trans1 = sitk.TranslationTransform(2)
    offset_x = moving_padding[0]
    offset_y = moving_padding[2]
    aff_trans1.SetOffset((offset_x, offset_y))
    
    aff_trans2 = sitk.TranslationTransform(2)
    offset_x = -moving_cropping[2]
    offset_y = -moving_cropping[0]
    aff_trans2.SetOffset((offset_x, offset_y))

    aff_trans3 = sitk.TranslationTransform(2)
    aff_trans3.SetOffset((fixed_cropping[2], fixed_cropping[0]))
    
    aff_trans4 = sitk.TranslationTransform(2)
    aff_trans4.SetOffset((-fixed_padding[0], -fixed_padding[2]))

    all_transforms.AddTransform(pre_downscale_transform)
    all_transforms.AddTransform(aff_trans4)
    all_transforms.AddTransform(aff_trans3)
    # all_transforms.AddTransform(displ_transform)
    all_transforms.AddTransform(transform)
    all_transforms.AddTransform(aff_trans2)
    all_transforms.AddTransform(aff_trans1)
    all_transforms.AddTransform(post_upscale_transform)
    return all_transforms


def compute_transforms(transformation: RegResult) -> Tuple[SimpleITK.SimpleITK.Transform, SimpleITK.SimpleITK.Transform]:
    forward_displacement_field = compose_reg_transforms(transformation)
    backward_displacement_field = compose_inv_reg_transforms(transformation)
    return forward_displacement_field, backward_displacement_field


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
                 options: Optional[RegistrationOptions] = None) -> 'RegistrationResult':
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
        return reg_result 

    def register_(self,
                 moving_img: numpy.array,
                 fixed_img: numpy.array,
                 moving_img_mask: Optional[numpy.array] = None,
                 fixed_img_mask: Optional[numpy.array] = None,
                 options: Optional[RegistrationOptions] = None,                  
                 **kwargs: Dict) -> 'RegistrationResult':
        # TODO: We only need a temp directory now, so get rid of output.
        if options is None:
            options = RegistrationOptions()
        # Step 1: Set up all the parameters and filenames as necessary.
        # Clean this up.
        path_temp = options.temporary_directory
        clean_if_exists(path_temp)
        path_temp, _ = derive_subdir(path_temp)
        create_if_not_exists(path_temp)
        path_output = join(path_temp, 'output')
        create_if_not_exists(path_output)
        path_output = join(path_output, 'registrations')
        path_output, subdir_num = derive_subdir(path_output)
        create_if_not_exists(path_output)
        # path_temp = args.get('tmp_dir', join(path_output, 'tmp'))
        self.path_temp = path_temp
        self.path_output = path_output
        affine_use_denoising = options.enable_affine_denoising
        deformable_use_denoising = options.enable_deformable_denoising
        # TODO: Implement autodownsampling if not set to get images to not bigger than 2000px
        pre_downsampling_factor = options.pre_downsampling_factor
        original_moving_image_size = moving_img.shape[:2]
        original_fixed_image_size = fixed_img.shape[:2]

        reg_params = {'s1': options.greedy_opts.s1,
                      's2': options.greedy_opts.s2,
                      'iteration_rigid': options.greedy_opts.iteration_rigid,
                      'resolution': options.resolution,
                      'affine_use_denoising': options.enable_affine_denoising,
                      'deformable_use_denoising': options.enable_deformable_denoising,
                      'options': options.to_dict(),
                      'pre_downsampling_factor': options.pre_downsampling_factor,
                      'original_moving_image_size': original_moving_image_size,
                      'original_fixed_image_size': original_fixed_image_size
                      }

        cmdln_returns = []
        # Convert to correct format, if necessary
        moving_img = correct_img_dtype(moving_img)
        fixed_img = correct_img_dtype(fixed_img)
        # moving_img = resample_by_factor(moving_img, pre_downsampling_factor)
        # fixed_img = resample_by_factor(fixed_img, pre_downsampling_factor)
        moving_img = resample_image_sitk(moving_img, pre_downsampling_factor)
        fixed_img = resample_image_sitk(fixed_img, pre_downsampling_factor)
        if moving_img_mask is None:
            moving_img_mask = self.segmentation_function(moving_img)
        if fixed_img_mask is None:
            fixed_img_mask = self.segmentation_function(fixed_img)
        # Do padding
        cropped_moving_mask, crop_params_mov = cropping(moving_img_mask)
        cropped_fixed_mask, crop_params_fix = cropping(fixed_img_mask)
        reg_params['cropping_params_mov'] = crop_params_mov
        reg_params['cropping_params_fix'] = crop_params_fix
        reg_params['original_shape_fixed_image'] = fixed_img.shape
        cropped_moving_img = moving_img[crop_params_mov[0]:crop_params_mov[1], crop_params_mov[2]:crop_params_mov[3]]
        cropped_fixed_img = fixed_img[crop_params_fix[0]:crop_params_fix[1], crop_params_fix[2]:crop_params_fix[3]]
        moving_pad, fixed_pad = get_symmetric_padding(cropped_moving_img, cropped_fixed_img)
        moving_img = pad_asym(cropped_moving_img, moving_pad)
        fixed_img = pad_asym(cropped_fixed_img, fixed_pad)
        moving_img_mask = pad_asym(cropped_moving_mask, moving_pad)
        fixed_img_mask = pad_asym(cropped_fixed_mask, fixed_pad)
        reg_params['moving_padding'] = moving_pad
        reg_params['fixed_padding'] = fixed_pad
        moving_img = apply_mask(moving_img, moving_img_mask)
        fixed_img = apply_mask(fixed_img, fixed_img_mask)
        _, cropping_padded_mov_mask = cropping(moving_img_mask)
        _, cropping_padded_fix_mask = cropping(fixed_img_mask)
        reg_params['cropped_padded_mov_mask'] = cropping_padded_mov_mask
        reg_params['cropped_padded_fix_mask'] = cropping_padded_fix_mask

        resample = options.resolution[0] / moving_img.shape[0] * 100
        smoothing = max(int(100 / (2 * resample)), 1) + 1
        reg_params['resample'] = resample
        reg_params['smoothing'] = smoothing

        requires_denoising = options.enable_affine_denoising or options.enable_deformable_denoising
        requires_standard_preprocessing = (not options.enable_affine_denoising) or (not options.enable_deformable_denoising)
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

        create_if_not_exists(path_metrics)
        create_if_not_exists(path_metrics_small_resolution)
        create_if_not_exists(path_metrics_full_resolution)

        # 3. Registration
        # Affine registration
        offset = int((height + (options.greedy_opts.kernel_size * 4)) / 10)
        reg_params['offset'] = offset
        reg_params['affine_iteration_vec'] = options.greedy_opts.affine_iteration_pyramid
        reg_params['deformable_iteration_vec'] = options.greedy_opts.nonrigid_iteration_pyramid

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

        if options.do_affine_registration:
            path_small_affine = os.path.join(path_metrics_small_resolution, 'small_affine.mat')
            aff_ret = affine_registration(self.path_to_greedy,
                                          fixed_img_path,
                                          moving_img_path,
                                          path_small_affine,
                                          offset,
                                          ia_init,
                                          options.greedy_opts
                                          )
            cmdln_returns.append(aff_ret)
        else:
            path_small_affine = None

        if options.do_nonrigid_registration:        

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
                                                         affine_pre_transform=path_small_affine,
                                                         ia=ia_init)
            # TODO: Improve error handling
            if deformable_reg_ret.returncode != 0:
                print(deformable_reg_ret.args)
                print(deformable_reg_ret.stderr)
                return
            cmdln_returns.append(deformable_reg_ret)
        else:
            path_small_warp = None
            path_small_warp_inv = None

        # Post processing
        # TODO: Most of these are not needed anymore.
        reg_params['WIDTH_small_fixed'] = current_fixed_preprocessed.width
        reg_params['HEIGHT_small_fixed'] = current_fixed_preprocessed.height
        reg_params['WIDTH_fixed_image_padded'] = current_fixed_preprocessed.width_padded
        reg_params['HEIGHT_fixed_image_padded'] = current_fixed_preprocessed.height_padded
        reg_params['WIDTH_fixed_image'] = current_fixed_preprocessed.width_original
        reg_params['HEIGHT_fixed_image'] = current_fixed_preprocessed.height_original
        reg_params['WIDTH_small_moving'] = current_moving_preprocessed.width
        reg_params['HEIGHT_small_moving'] = current_moving_preprocessed.height
        reg_params['WIDTH_moving_image_padded'] = current_moving_preprocessed.width_padded
        reg_params['HEIGHT_moving_image_padded'] = current_moving_preprocessed.height_padded
        reg_params['WIDTH_moving_image'] = current_moving_preprocessed.width_original
        reg_params['HEIGHT_moving_image'] = current_moving_preprocessed.height_original
        
        # Write small ref image to file for warping of coordinates
        small_fixed = sitk.ReadImage(current_fixed_preprocessed.image_path)
        empty_fixed_img = small_fixed[:,:]
        empty_fixed_img[:,:] = 0
        path_to_small_ref_image = join(path_output, 'small_ref_image.nii.gz')
        sitk.WriteImage(empty_fixed_img, path_to_small_ref_image)


        factor = 100 / resample
        reg_params['factor'] = factor

        # If no non-rigid registration is performed we keep the affine transform unbounded.
        # TODO: This needs some changing. There should be an option not to composite affine and nonrigid registrations, so that affine keeps being unbounded.
        if options.do_nonrigid_registration and not options.keep_affine_transform_unbounded:
            path_small_composite_warp = os.path.join(path_metrics_small_resolution, 'small_composite_warp.nii.gz')
            composite_warps(
                self.path_to_greedy,
                path_small_affine,
                path_small_warp,
                path_to_small_ref_image,
                path_small_composite_warp            
            )
            path_big_composite_warp = os.path.join(path_metrics_full_resolution, 'big_composite_warp.nii.gz')
            rescale_warp(
                path_small_composite_warp,
                path_big_composite_warp,
                (current_fixed_preprocessed.width, current_fixed_preprocessed.height),
                (current_fixed_preprocessed.width_original,
                 current_fixed_preprocessed.height_original),
                factor)
            path_small_inverted_composite_warp = os.path.join(path_metrics_small_resolution, 'small_inv_composite_warp.nii.gz')
            composite_warps(
                self.path_to_greedy,
                path_small_affine,
                path_small_warp_inv,
                path_to_small_ref_image,
                path_small_inverted_composite_warp,
                invert=True 
            )
            path_big_composite_warp_inv = os.path.join(path_metrics_full_resolution, 'big_inv_composite_warp.nii.gz')
            rescale_warp(
                path_small_inverted_composite_warp,
                path_big_composite_warp_inv,
                (current_moving_preprocessed.width, current_moving_preprocessed.height),
                (current_moving_preprocessed.width_original,
                 current_moving_preprocessed.height_original),
                factor)

            displacement_field = sitk.ReadImage(path_big_composite_warp, sitk.sitkVectorFloat64)
            rotated_displ_field = sitk.GetArrayFromImage(displacement_field)
            rotated_displ_field *= -1
            rotated_displ_field_sitk = sitk.GetImageFromArray(rotated_displ_field, True)
            displ_field = sitk.Image(rotated_displ_field_sitk) 
            displ_field = sitk.Cast(displ_field, sitk.sitkVectorFloat64)
            forward_transform = sitk.DisplacementFieldTransform(2)
            forward_transform.SetDisplacementField(displ_field)
            
            inv_displacement_field = sitk.ReadImage(path_big_composite_warp_inv, sitk.sitkVectorFloat64)
            rotated_displ_field = sitk.GetArrayFromImage(inv_displacement_field)
            rotated_displ_field *= -1
            rotated_displ_field_sitk = sitk.GetImageFromArray(rotated_displ_field, True)
            displ_field = sitk.Image(rotated_displ_field_sitk) 
            displ_field = sitk.Cast(displ_field, sitk.sitkVectorFloat64)
            backward_transform = sitk.DisplacementFieldTransform(2)
            backward_transform.SetDisplacementField(displ_field)
        elif options.do_nonrigid_registration and options.keep_affine_transform_unbounded:
            # First rescale affine transforms
            forward_affine_transform = rescale_affine_2(path_small_affine, factor)
            backward_affine_transform = forward_affine_transform.GetInverse()
            
            path_big_warp = os.path.join(path_metrics_full_resolution, 'big_warp.nii.gz')
            rescale_warp(
                path_small_warp,
                path_big_warp,
                (current_fixed_preprocessed.width, current_fixed_preprocessed.height),
                (current_fixed_preprocessed.width_original,
                 current_fixed_preprocessed.height_original),
                factor)
            
            path_big_warp_inv = os.path.join(path_metrics_full_resolution, 'big_inv_warp.nii.gz')
            rescale_warp(
                path_small_warp_inv,
                path_big_warp_inv,
                (current_fixed_preprocessed.width, current_fixed_preprocessed.height),
                (current_fixed_preprocessed.width_original,
                 current_fixed_preprocessed.height_original),
                factor)

            displacement_field = sitk.ReadImage(path_big_warp, sitk.sitkVectorFloat64)
            rotated_displ_field = sitk.GetArrayFromImage(displacement_field)
            rotated_displ_field *= -1
            rotated_displ_field_sitk = sitk.GetImageFromArray(rotated_displ_field, True)
            displ_field = sitk.Image(rotated_displ_field_sitk) 
            displ_field = sitk.Cast(displ_field, sitk.sitkVectorFloat64)
            forward_deformable_transform = sitk.DisplacementFieldTransform(2)
            forward_deformable_transform.SetDisplacementField(displ_field)
            
            inv_displacement_field = sitk.ReadImage(path_big_warp_inv, sitk.sitkVectorFloat64)
            rotated_displ_field = sitk.GetArrayFromImage(inv_displacement_field)
            rotated_displ_field *= -1
            rotated_displ_field_sitk = sitk.GetImageFromArray(rotated_displ_field, True)
            displ_field = sitk.Image(rotated_displ_field_sitk) 
            displ_field = sitk.Cast(displ_field, sitk.sitkVectorFloat64)
            backward_deformable_transform = sitk.DisplacementFieldTransform(2)
            backward_deformable_transform.SetDisplacementField(displ_field)
                
            forward_transform = sitk.CompositeTransform(2)
            forward_transform.AddTransform(forward_affine_transform)
            forward_transform.AddTransform(forward_deformable_transform)
            
            backward_transform = sitk.CompositeTransform(2)
            backward_transform.AddTransform(backward_deformable_transform)
            backward_transform.AddTransform(backward_affine_transform)

            path_small_composite_warp = ''
            path_small_inverted_composite_warp = ''
            path_big_composite_warp = ''
            path_big_composite_warp_inv = ''

        else:
            # Set some paths to empty. Will remove them entirely later.
            path_small_composite_warp = ''
            path_small_inverted_composite_warp = ''
            path_big_composite_warp = ''
            path_big_composite_warp_inv = ''
            # TODO: Do I need to -1 this transformation? (Because the direction was different for the displacement field. Though things seem to work with
            # affine transformation.) Check!!
            forward_transform = rescale_affine_2(path_small_affine, factor)
            backward_transform = forward_transform.GetInverse()
            # backward_transform = invert_affine_transform(forward_transform)

        # Check those 2

        # reg_param_outpath = os.path.join(path_output, 'reg_params.json')
        # with open(reg_param_outpath, 'w') as f:
        #     json.dump(reg_params, f)


        displacement_field = None
        inv_displacement_field = None
        reg_result = RegResult(
            path_to_small_moving=current_moving_preprocessed.image_path,
            path_to_small_fixed=current_fixed_preprocessed.image_path,
            path_to_small_composite=path_small_composite_warp,
            path_to_big_composite=path_big_composite_warp,
            path_to_small_inv_composite=path_small_inverted_composite_warp,
            path_to_big_inv_composite=path_big_composite_warp_inv,
            cmdl_log=cmdln_returns,
            reg_params=reg_params,
            path_to_small_ref_image=path_to_small_ref_image,
            sub_dir_key=subdir_num,
            displacement_field=displacement_field,
            inv_displacement_field=inv_displacement_field
        )

        # TODO: For affine transforms: Write function to convert all 5 transforms into a single transform.
        # TODO: Add option to compress all 5 transforms into a single displacement field.
        composited_forward_transform = compose_reg_transforms(forward_transform, reg_result)
        composited_backward_transform = compose_inv_reg_transforms(backward_transform, reg_result)
        fixed_transform = GFHTransform(original_fixed_image_size, composited_forward_transform)
        moving_transform = GFHTransform(original_moving_image_size, composited_backward_transform)
        registration_result = RegistrationResult(fixed_transform=fixed_transform, moving_transform=moving_transform, cmdln_returns=cmdln_returns)
        # Return this!
        
        if options.remove_temporary_directory:
            self.__cleanup_temporary_directory(path_temp)        
        return registration_result

    def groupwise_registration(self,
                               image_mask_list: List[Tuple[numpy.array, Optional[numpy.array]]],
                               affine_options: Optional[RegistrationOptions] = None,
                               nonrigid_option: Optional[RegistrationOptions] = None,
                               skip_deformable_registration: bool = False,
                               ):
        if affine_options is None:
            affine_options = RegistrationOptions()
            affine_options.do_nonrigid_registration = False
        if nonrigid_option is None:
            nonrigid_option = RegistrationOptions()
        nonrigid_option.do_affine_registration = False
        # Stage1: Affine register along the the sequence.
        moving_image, moving_mask = image_mask_list[0]
        # TODO: Do i really need to do that?
        # Need masks only for the deformable registration part, since the moving images are pretransformed there.
        # if moving_mask is None:
        #     seg_fun = load_yolo_segmentation()
        #     moving_mask = seg_fun(moving_image)
        # warped_image = moving_image.copy()
        # warped_mask = moving_mask.copy()
        # Need to get mask is None supplied
        # First do affine swipe
        registered_images = []
        affine_transform_lists = []
        for (fixed_image, fixed_mask) in image_mask_list[1:]:
            # This kind of makes transformations that are all registered to the original fixed image. Consider having them only be pairwise.
            sub_options = RegistrationOptions()
            sub_options.keep_affine_transform_unbounded = True
            sub_options.do_nonrigid_registration = False
            reg_result = self.register_(moving_image, fixed_image, moving_mask, fixed_mask, sub_options)
            affine_transform_lists.append(reg_result)
            # map_to_transforms[rev_idx] = (forward, backward)
            moving_image = fixed_image
            moving_mask = fixed_mask
            # warped_image = self.transform_image_(warped_image, reg_result.fixed_transform.transform, reg_result.fixed_transform.size, 'LINEAR')
            # warped_mask = self.transform_image_(warped_mask, reg_result.fixed_transform.transform, reg_result.fixed_transform.size, 'NN')
            # registered_images.append(warped_image)
        # Stage 2: Take the matched images and do a nonrigid registration
        if skip_deformable_registration:
            g_res = GroupwiseRegResult(affine_transform_lists, [])
            return g_res, registered_images
        nonrigid_transformations = []
        nonrigid_warped_images = []
        fixed_image, fixed_mask = image_mask_list[-1]
        for idx, (moving_image, moving_mask) in enumerate(image_mask_list[:-1]):
            if moving_mask is None:
                moving_mask = self.segmentation_function(moving_image)
            composited_fixed_transform = compose_transforms([x.fixed_transform for x in affine_transform_lists][idx:])
            # composited_moving_transform = compose_transforms([x.moving_transform for x in transform_lists][idx:][::-1])
            warped_image = self.transform_image(moving_image, composited_fixed_transform, 'LINEAR')
            warped_mask = self.transform_image(moving_mask, composited_fixed_transform, 'NN')
            nonrigid_reg_result = self.register(warped_image, fixed_image, warped_mask, fixed_mask, options=nonrigid_option)
            deformable_warped_image = self.transform_image(warped_image, nonrigid_reg_result.fixed_transform, 'LINEAR')
            # Do deformable mask?
            nonrigid_warped_images.append(deformable_warped_image)
            nonrigid_transformations.append(nonrigid_reg_result)
        groupwise_registration_results = GroupwiseRegResult(affine_transform_lists, nonrigid_transformations)
        return groupwise_registration_results, nonrigid_warped_images

    def transform_image(self,
                        image: numpy.array,
                        transform: 'GFHTransform',
                        interpolation_mode: str = 'LINEAR') -> numpy.array:
        return self.transform_image_(image,
                                     transform.transform,
                                     transform.size,
                                     interpolation_mode)

    def transform_image_(self,
                        image: numpy.array, 
                        transform: SimpleITK.SimpleITK.Transform,
                        size: Tuple[int,int],
                        interpolation_mode: str = 'LINEAR') -> numpy.array:
        interpolator = sitk.sitkLinear if interpolation_mode == 'LINEAR' else sitk.sitkNearestNeighbor
        # size = transformation_matrix.GetSize()
        ref_img = sitk.GetImageFromArray(np.zeros((size[0], size[1])), True)
        sitk_image = sitk.GetImageFromArray(image, True)
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(ref_img)
        resampler.SetInterpolator(interpolator)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(transform)
        warped_image_sitk = resampler.Execute(sitk_image)
        registered_image = sitk.GetArrayFromImage(warped_image_sitk)
        return registered_image

    def transform_pointset(self,
                           pointset: numpy.array,
                           transform: GFHTransform) -> numpy.array:
        return self.transform_pointset_(
            pointset,
            transform.transform
        )

    def transform_pointset_(self,
                         pointset: numpy.array,
                         transform: SimpleITK.SimpleITK.Transform) -> numpy.array:
        """Transform pointset. 
    
        Args:
            pointset (pandas.DataFrame): Supplied as a dataframe. x and y columns are used for transforming data.
            transformation (RegResult): 
            args (_type_, optional): Any additional arguments for transforming. Defaults to None.
    
        Returns:
            Any: _description_
        """
        pointset -= 0.5
        warped_points = []
        for i in range(pointset.shape[0]):
            point = (pointset[i,0], pointset[i,1])
            warped_point = transform.TransformPoint(point)
            warped_points.append(warped_point)
        warped_pointset = np.array(warped_points)
        return warped_pointset

    def transform_geojson(self,
                          geojson_data: geojson.GeoJSON,
                          transformation: SimpleITK.SimpleITK.Image,
                          **kwards) -> Any:
        if not isinstance(geojson_data, list):
            geometries = geojson_data['features']
        else:
            geometries = geojson_data
        warped_geometries = []
        for _, geometry in enumerate(geometries):
            warped_geometry = geojson.utils.map_tuples(lambda coords: self.__warp_geojson_coord_tuple(coords, transformation), geometry)
            warped_geometries.append(warped_geometry)
        if not isinstance(geojson_data, list):
            geojson_data['features'] = warped_geometries
            return geojson_data
        else:
            return warped_geometries

    def __warp_geojson_coord_tuple(self, coord, transform):
        ps = np.array([[coord[0], coord[1]]]).astype(float)
        warped_ps = self.transform_pointset(ps, transform)
        return (warped_ps[0, 0], warped_ps[0, 1])

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
