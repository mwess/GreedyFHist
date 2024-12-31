"""GreedyFHist, the registration algorithm, includes pairwise/groupwise registration, transform for various file formats. 

This module handles the core registration functionality via the 
GreedyFHist class. Results are exported as GroupwiseRegResult or 
RegistrationTransforms. 
"""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
import json
import os
from os.path import join, exists
from pathlib import Path
import shutil
import subprocess
from typing import Any, Optional

import geojson
import numpy, numpy as np
import SimpleITK, SimpleITK as sitk
import tqdm

from greedyfhist.registration.result_types import (
    GroupwiseRegResult,
    GFHTransform,
    RegistrationResult,
    RegistrationTransforms,
    InternalRegParams,
    PreprocessedData,
    compose_transforms,
    compose_registration_results
)


from greedyfhist.utils.io import (
    create_if_not_exists, 
    write_mat_to_file, 
    clean_if_exists, 
    read_image,
    write_to_ometiffile,
    affine_transform_to_file,
    derive_subdir
)

from greedyfhist.utils.image import (
    rescale_warp,
    rescale_affine,
    denoise_image,
    apply_mask,
    com_affine_matrix,
    pad_image,
    resample_image_with_gaussian,
    pad_asym,
    get_symmetric_padding,
    cropping,
    resample_image_sitk,
    derive_resampling_factor,
    realign_displacement_field,
    read_affine_transform
)

from greedyfhist.utils.utils import (
    deformable_registration, 
    affine_registration, 
    composite_warps, 
    affine_registration, 
    deformable_registration,
    compose_reg_transforms,
    compose_inv_reg_transforms,
    correct_img_dtype,
    derive_sampling_factors
)

from greedyfhist.segmentation import load_yolo_segmentation
from greedyfhist.options import RegistrationOptions, PreprocessingOptions
from greedyfhist.utils.tiling import (
    reassemble_sitk_displacement_field, 
    ImageTile, 
    extract_image_tiles, 
    get_tile_params
)

def preprocessing(image: numpy.ndarray,
                  preprocessing_options: PreprocessingOptions,
                  resolution: tuple[int, int],
                  kernel_size: int,
                  tmp_path: str,
                  skip_denoising: bool = False
                  ) -> 'PreprocessedData':
    """Image preprocessing applied after images have transformed to a uniform shape. 
    Preprocessing steps are: (1) Denoising, (2) Gaussian smoothing, (3) Downsampling, 
    (4) Grayscale conversion, (5) Padding.

    Args:
        image (numpy.ndarray): Source image to be preprocessed.
        preprocessing_options (PreprocessingOptions): Contains parameters for mean shift filtering.
        resolution (tuple[int, int]): Target downscaling resolution
        kernel_size (int): Size of kernel used for padding after downscaling
        tmp_path (str): Temporary directory used for storing the preprocessed image.
        skip_denoising (bool, optional): Another toggle to enable/disable mean shift filtering. Defaults to False.

    Returns:
        PreprocessedData: Preprocessed image data.
    """
    create_if_not_exists(tmp_path)
    if preprocessing_options.enable_denoising and not skip_denoising:
        image = denoise_image(image,
                              sp=preprocessing_options.moving_sr,
                              sr=preprocessing_options.moving_sp)
    tmp_dir = join(tmp_path, 'preprocessed')
    create_if_not_exists(tmp_dir)
    resample = resolution[0] / image.shape[0] * 100
    smoothing = max(int(100 / (2 * resample)), 1) + 1    
    image_preprocessed = preprocess_image_for_greedy(image,
                                                     kernel_size,
                                                     resolution,
                                                     smoothing,
                                                     tmp_dir)
    return image_preprocessed


def preprocess_image_for_greedy(image: numpy.ndarray,
                     kernel: int,
                     resolution: tuple[int, int],
                     smoothing: int,
                     tmp_dir: str) -> 'PreprocessedData':
    """Performs final preprocessing steps of image and saves image for 
    greedy under the filename is 'new_small_image.nii.gz' in the 
    provided tmp_dir.


    Args:
        image (numpy.ndarray):
        kernel (int): kernel size
        resolution (Tuple[int, int]): resolution after downscaling
        smoothing (int): Gaussian smoothing applied for preventing 
                         anti-aliasing.
        tmp_dir (str): Directory for storing.

    Returns:
        PreprocessedData: Contains path to downscaled image and 
                          additional image parameters.
    """
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


def reassemble_to_gfh_transform(displ_tiles: list[ImageTile], 
                                trx_shape: tuple[int, int] | tuple[int, int, int]) -> GFHTransform:
    """Reassembles transformations for each tile into one transformation.

    Args:
        displ_tiles (list[ImageTile]): 
        trx_shape (tuple[int, int] | tuple[int, int, int]): 

    Returns:
        GFHTransform: 
    """
    displ_np = reassemble_sitk_displacement_field(displ_tiles, trx_shape)
    displ_sitk = sitk.GetImageFromArray(displ_np, True)
    displ_sitk = sitk.Cast(displ_sitk, sitk.sitkVectorFloat64)
    trx = sitk.DisplacementFieldTransform(displ_sitk)
    gfh_transform = GFHTransform(trx_shape, trx)
    return gfh_transform


@dataclass
class GreedyFHist:
    """
    Registration class. Performs registrations and transformation for paiwise images and groupwise images.

    Attributes
    ----------

    name: str
        Identifier

    path_to_greedy: str
        Path to greedy executable. Not needed if Greedy is on PATH.

    segmentation_function: Optional[Callable]
        Segmentation function for foreground segmentation.

    """

    name: str = 'GreedyFHist'
    path_to_greedy: str = 'greedy'
    use_docker_container: bool = False
    segmentation_function: Callable | None = None
    cmdln_returns: list[Any] = field(default_factory=lambda: [])

    def __post_init__(self):
        if self.segmentation_function is None:
            self.segmentation_function = load_yolo_segmentation()

    def register_from_filepaths(self,
                                moving_img_path: str,
                                fixed_img_path: str,
                                target_img_path: str,
                                moving_img_mask_path: str | None = None,
                                fixed_img_mask_path: str | None = None,
                                options: RegistrationOptions | None = None,
                                transform_path: str | None = False
                                ) -> tuple['RegistrationResult', numpy.ndarray | None]:
        """Register two images by supplying only filepaths. Optionally filepaths for masks can be supplied.
        If images cannot be read for some reason, consider the method `register`.
        
        Args:
            moving_img_path (str): Path to source image.
            fixed_img_path (str): Path to target image.
            moving_img_mask (Optional[numpy.ndarray], optional): Optional moving mask. Is otherwise derived automatically. Defaults to None.
            fixed_img_mask (Optional[numpy.ndarray], optional): Optional fixed mask. Is otherwise dervied automatically. Defaults to None.
            options (Optional[Options], optional): Can be supplied. Otherwise default arguments are used. Defaults to None.

        Returns:
            RegistrationResult: Computed registration result.        
        """
        moving_image, mov_metadata = read_image(moving_img_path)
        fixed_image, fix_metadata = read_image(fixed_img_path)
        if moving_img_mask_path is not None:
            moving_img_mask, _ = read_image(moving_img_mask_path, True)
        else:
            moving_img_mask = None
        if fixed_img_mask_path is not None:
            fixed_img_mask, _ = read_image(fixed_img_mask_path, True)
        else:
            fixed_img_mask = None
        registration_result = self.register(moving_image,
                                            fixed_image,
                                            moving_img_mask,
                                            fixed_img_mask,
                                            options)
        
        warped_image = self.transform_image(moving_image, registration_result.registration.forward_transform)
        write_to_ometiffile(warped_image,
                            target_img_path,
                            metadata=fix_metadata)
        if transform_path is not None:
            registration_result.to_directory(transform_path)
        return registration_result, warped_image

    def groupwise_registration_from_filepaths(self,
                               image_mask_filepaths: list[tuple[str, str | None]],
                               target_directory: str,
                               options: RegistrationOptions | None = None,
                               ) -> tuple[GroupwiseRegResult, list[numpy.ndarray]]:
        """Performs groupwise registration based on filepaths. 

        Args:
            image_mask_filepaths (list[tuple[str, str  |  None]]): _description_
            target_directory (str): Target directory for storing registered images and transformations.
            options (RegistrationOptions | None, optional): Options for registration. Defaults to None.

        Returns:
            tuple[GroupwiseRegResult, list[numpy.ndarray]]: Transformations and transformed images.
        """
        img_mask_list = []
        img_paths = []
        for paths in image_mask_filepaths:
            if isinstance(paths, tuple) or isinstance(paths, list):
                img_path = paths[0]
                mask_path = paths[1]
            else:
                img_path = paths
                mask_path = None
            img_paths.append(img_path)
            img, _ = read_image(img_path)
            if mask_path is not None:
                mask, _ = read_image(mask_path, True).squeeze()
            else:
                mask = None
            img_mask_list.append((img, mask))
        groupwise_registration_result, warped_images = self.groupwise_registration(img_mask_list, options)
        warped_images.append(img_mask_list[-1][0])
        image_target_directory = join(target_directory, 'images')
        Path(image_target_directory).mkdir(parents=True, exist_ok=True)
        for img, img_path in zip(warped_images, img_paths):
            fname = os.path.basename(img_path)
            if fname.endswith('.ome.tif'):
                fname = fname.replace('.ome.tif', '')
            else:
                fname = os.path.splitext(fname)[0]
            fname = os.path.join(image_target_directory, f'{fname}.ome.tif')
            write_to_ometiffile(img, fname)
        groupwise_registration_result.to_file(join(target_directory, 'transform'))
        return groupwise_registration_result, warped_images


    def register_tiles(self,
                       moving_image_tiles: list[ImageTile],
                       fixed_image_tiles: list[ImageTile],
                       tile_reg_opts: RegistrationOptions | None = None,
                       verbose: bool = False) -> RegistrationResult:
        """Register moving and fixed image tiles. Each tile in `moving_image_tiles` is registered
        with a corresponding tile at the same position as `fixed_image_tiles`. 

        Args:
            moving_image_tiles (ImageTile): List of moving image tiles.
            fixed_image_tiles (ImageTile): List of fixed image tiles.
            tile_reg_opts (RegistrationOptions | None, optional): Nonrigid registration options applied to register pairs
            of tiles. Defaults to None.
            verbose (bool, optional): Defaults to False.

        Returns:
            RegistrationResult:
        """
        fw_displ_tiles = []
        bw_displ_tiles = []
        tile_transforms = []
        for idx, (moving_image_tile_, fixed_image_tile_) in tqdm.tqdm(enumerate(zip(moving_image_tiles, fixed_image_tiles)), disable=not verbose):
            if tile_reg_opts is None:
                nrrt_options = RegistrationOptions()
                nrrt_options.do_affine_registration = False
                # nrrt_options.remove_temporary_directory = False
                # nrrt_options.temporary_directory = f'tmp_{idx}'
            else:
                nrrt_options = tile_reg_opts

            if verbose:
                print(nrrt_options)
            moving_image_ = moving_image_tile_.image
            fixed_image_ = fixed_image_tile_.image
            mask = np.ones(moving_image_.shape[:2])

            try:
                registration_result_ = self.register(moving_img=moving_image_,
                                                            fixed_img=fixed_image_,
                                                            moving_img_mask=mask,
                                                            fixed_img_mask=mask,
                                                            options=nrrt_options)
                # Foward transform
                transform_to_displacementfield = sitk.TransformToDisplacementField(
                    registration_result_.registration.forward_transform.transform,
                    outputPixelType=sitk.sitkVectorFloat64,
                    size=registration_result_.registration.forward_transform.size[::-1]
                )
                fw_displ_tile = ImageTile(transform_to_displacementfield,
                                    fixed_image_tile_.x_props,
                                    fixed_image_tile_.y_props,
                                    
                )
                # Backward transform
                transform_to_displacementfield = sitk.TransformToDisplacementField(
                    registration_result_.registration.backward_transform.transform,
                    outputPixelType=sitk.sitkVectorFloat64,
                    size=registration_result_.registration.backward_transform.size[::-1]
                )
                bw_displ_tile = ImageTile(transform_to_displacementfield,
                                    fixed_image_tile_.x_props,
                                    fixed_image_tile_.y_props
                )        
            except Exception as e:
                if verbose:
                    print(f'Something went wrong in tile: {idx}')
                    print(e)
                # TODO: Fix this! Needs to be a transform.
                fw_tile = sitk.GetImageFromArray(np.dstack((mask.astype(np.float64), mask.astype(np.float64))), True)
                fw_tile = sitk.Cast(fw_tile, sitk.sitkVectorFloat64)
                fw_displ_tile = ImageTile(fw_tile,
                                    fixed_image_tile_.x_props,
                                    fixed_image_tile_.y_props)
                bw_tile = sitk.GetImageFromArray(np.dstack((mask.astype(np.float64), mask.astype(np.float64))), True)
                bw_tile = sitk.Cast(bw_tile, sitk.sitkVectorFloat64)
                bw_displ_tile = ImageTile(bw_tile,
                                    moving_image_tile_.x_props,
                                    moving_image_tile_.y_props)
                
            fw_displ_tiles.append(fw_displ_tile)
            bw_displ_tiles.append(bw_displ_tile)
        fixed_transform = reassemble_to_gfh_transform(fw_displ_tiles, fixed_image_tiles[0].original_shape)
        moving_transform = reassemble_to_gfh_transform(bw_displ_tiles, moving_image_tiles[0].original_shape)
        registration_transforms = RegistrationTransforms(forward_transform=fixed_transform, backward_transform=moving_transform)
        rev_registration_transforms = RegistrationTransforms(forward_transform=moving_transform, backward_transform=fixed_transform)
        return RegistrationResult(registration_transforms, rev_registration_transforms)  


    def nrpt_registration(self,
                          moving_image: numpy.ndarray,
                          fixed_image: numpy.ndarray,
                          registration_options: RegistrationOptions,
                          verbose: bool = False) -> RegistrationResult:
        """Pyramidical tiling nonrigid registration (ptnr) mode. A registration modus that can be used in addition, or instead
        of nonrigid registration. 
        
        The base idea behind ptnr is that images are divided into tiles which can then be registered one-by-one due to the locality of 
        nonrigid registration. After tiles have been registered, the resulting deformation fields are stitched together.
        
        nrpt is pyramidical with each layer increasing the number of tiles to approximate the most accurate registration as close as possible.
        By default, pyramids grow quadratically, though other options exist.
        
        Note: Images are expected to be of the same shape. Performing an affine or nonrigid registration beforehand guarantees that both
        input images have the same shape.

        Args:
            moving_image (numpy.ndarray): 
            fixed_image (numpy.ndarray): 
            registration_options (RegistrationOptions): 
            verbose (bool, optional): Defaults to False.

        Returns:
            RegistrationResult:
        """
        registration_options.do_affine_registration = False
        nrpt_opts = registration_options.nrpt_options
        return self.nrpt_registration_(
            moving_image,
            fixed_image,
            stop_condition_tile_resolution=nrpt_opts.stop_condition_tile_resolution,
            stop_condition_pyramid_counter=nrpt_opts.stop_condition_pyramid_counter,
            max_pyramid_depth=nrpt_opts.max_pyramid_depth,
            pyramid_resolutions=nrpt_opts.pyramid_resolutions,
            pyramid_tiles_per_axis=nrpt_opts.pyramid_tiles_per_axis,
            nrpt_tile_options=registration_options,
            tile_overlap=nrpt_opts.tile_overlap,
            verbose=verbose
        )


    def nrpt_registration_(self,
                        moving_image: numpy.ndarray,
                        fixed_image: numpy.ndarray,
                        stop_condition_tile_resolution: bool = True,
                        stop_condition_pyramid_counter: bool = False,
                        max_pyramid_depth: int | None = None,
                        pyramid_resolutions: list[int] | None = None,
                        pyramid_tiles_per_axis: list[int] | None = None,
                        nrpt_tile_options: RegistrationOptions | None = None,
                        tile_overlap: float = 0.75,
                        verbose: bool = False) -> RegistrationResult:
        """Pyramidical tiling nonrigid registration (ptnr) mode. A registration modus that can be used in addition, or instead
        of nonrigid registration. 
        
        The base idea behind ptnr is that images are divided into tiles which can then be registered one-by-one due to the locality of 
        nonrigid registration. After tiles have been registered, the resulting deformation fields are stitched together.
        
        nrpt is pyramidical with each layer increasing the number of tiles to approximate the most accurate registration as close as possible.
        By default, pyramids grow quadratically, though other options exist.
        
        Note: Images are expected to be of the same shape. Performing an affine or nonrigid registration beforehand guarantees that both
        input images have the same shape.

        Args:
            moving_image (numpy.ndarray): Source image.
            fixed_image (numpy.ndarray): Target image.
            stop_condition_tile_resolution (bool, optional): If True, stops the pyramid once the size of tiles is smaller than that of 
                resampled images in GreedyFHist's preprocessing. Defaults to True.
            stop_condition_pyramid_counter (bool, optional): If True, stops the pyramid pyramid after `max_pyramid_depth` has been reached. 
                `stop_condition_tile_resolution` has precedence. Defaults to False.
            max_pyramid_depth (int | None, optional): Maximum depth of pyramid. Only used if `stop_condition_pyramid_counter` is True. Defaults to None.
            pyramid_resolutions (list[int] | None, optional): Tile resolutions used at each pyramid level. Requires that `stop_condition_pyramid_counter`
            is True and `max_pyramid_depth` is None. Uses list length as maximum depth. Defaults to None.
            pyramid_tiles_per_axis (list[int] | None, optional): Tiles per axis for each pyramid. Requires that `stop_condition_pyramid_counter`
            is True and `max_pyramid_depth` is None. Uses list length as maximum depth. Defaults to None.
            nrpt_tile_options (RegistrationOptions | None, optional): Registration options passed to each tile registration. If None,
                default options are used. Defaults to None.
            tile_overlap (float, optional): Overlap between two neighboring tiles. Needed to avoid false registration along edges. Discarded during stitching. Defaults to 0.75.
            verbose (bool, optional): Prints more information. Defaults to False.

        Returns:
            RegistrationResult: _description_
        """
        
        if nrpt_tile_options is None: 
            nrpt_tile_options = RegistrationOptions()
        
        nrpt_tile_options.do_affine_registration = False
        
        temp_image = moving_image.copy()

        # Should it start with 0 or not. 
        loop_counter = 1

        reg_results = []
        temp_images = [temp_image]

        actual_tile_size = np.inf

        downsampling_size = nrpt_tile_options.nonrigid_registration_options.resolution[0]

        if stop_condition_pyramid_counter and stop_condition_tile_resolution:
            stop_condition_tile_resolution = True
            stop_condition_pyramid_counter = False

        if stop_condition_pyramid_counter and max_pyramid_depth is None:
            max_pyramid_depth = np.inf
            if pyramid_tiles_per_axis is not None:
                max_pyramid_depth = len(pyramid_tiles_per_axis)
            if pyramid_resolutions is not None:
                max_pyramid_depth = min(max_pyramid_depth, len(pyramid_resolutions))
            if max_pyramid_depth == np.inf:
                raise Exception('Passed arguments are incompatible.')
        idx = 0
        while True:
            if verbose:
                print(f'Next loop: {loop_counter}')
            
            if stop_condition_pyramid_counter and loop_counter > max_pyramid_depth:
                if verbose:
                    print('Aborting loop because maximum pyramid depth has been reached.')
                break

            ptnr_opts = deepcopy(nrpt_tile_options)
            
            if pyramid_resolutions is not None:
                res = pyramid_resolutions[idx]
                nrpt_tile_options.nonrigid_registration_options.resolution = (res, res)
                
            # If loop_counter == 0 just do a normal nonrigid registration, since this will at least ensure
            # that both images have the same shape.
            
            
            n_tiles = int(2**loop_counter) if pyramid_tiles_per_axis is None else pyramid_tiles_per_axis[idx]
            if verbose:
                print(f'Number of tiles: {n_tiles}')
            x_tile_props = get_tile_params(temp_image.shape[0], n_tiles = n_tiles, overlap = tile_overlap)
            y_tile_props = get_tile_params(temp_image.shape[1], n_tiles = n_tiles, overlap = tile_overlap)    
            
            actual_tile_size = x_tile_props[4][0] - x_tile_props[1][0]
            if verbose:
                print(f'Actual tile size: {actual_tile_size}')
            if actual_tile_size < downsampling_size and stop_condition_tile_resolution:
                if verbose:
                    print('Aborting loop because minimal tile size has been reached.')
                break
            
            moving_image_tiles = extract_image_tiles(temp_image, x_tile_props, y_tile_props)
            fixed_image_tiles = extract_image_tiles(fixed_image, x_tile_props, y_tile_props)
            
            if verbose:
                print('Tiles extracted.')
                
            ptnr_reg = self.register_tiles(moving_image_tiles, fixed_image_tiles, ptnr_opts, verbose)
            reg_results.append(ptnr_reg)
            
            if verbose:
                print('Tiles registered.')
            temp_image = self.transform_image(temp_image, ptnr_reg.registration.forward_transform, 'LINEAR')
            temp_images.append(temp_image)
            
            loop_counter += 1    
        final_reg_result = compose_registration_results(reg_results)
        return final_reg_result

    def register(self,
                 moving_img: numpy.ndarray,
                 fixed_img: numpy.ndarray,
                 moving_img_mask: numpy.ndarray | None = None,
                 fixed_img_mask: numpy.ndarray | None = None,
                 options: RegistrationOptions | None = None,
                 verbose: bool = False) -> 'RegistrationResult':
        """Performs pairwise registration from moving_img to fixed_img. Optional tissue masks can be provided.
        Options are supplied via the options arguments.

        Documentation of the registration algorithm can be found here: ...
        

        Args:
            moving_img (numpy.ndarray): Source image.
            fixed_img (numpy.ndarray): Target image.
            moving_img_mask (Optional[numpy.ndarray], optional): Optional moving mask. Is otherwise derived automatically. Defaults to None.
            fixed_img_mask (Optional[numpy.ndarray], optional): Optional fixed mask. Is otherwise dervied automatically. Defaults to None.
            options (Optional[Options], optional): Can be supplied. Otherwise default arguments are used. Defaults to None.
            verbose (bool): Prints out more information. Defaults to False.

        Returns:
            RegistrationResult: Contains computed registration result.
        """
        if options is None:
            options = RegistrationOptions()
        reg_result = self.register_(moving_img,
                                    fixed_img,
                                    moving_img_mask,
                                    fixed_img_mask,
                                    options)
        return reg_result 

    def register_(self,
                 moving_img: numpy.ndarray,
                 fixed_img: numpy.ndarray,
                 moving_img_mask: numpy.ndarray | None = None,
                 fixed_img_mask: numpy.ndarray | None = None,
                 options: RegistrationOptions | None = None,                  
                 **kwargs: dict) -> tuple['RegistrationTransforms', Optional['RegistrationTransforms']]:
        """Computes registration from moving image to fixed image.

        Args:
            moving_img (numpy.ndarray):
            fixed_img (numpy.ndarray): 
            moving_img_mask (Optional[numpy.ndarray], optional): Optional moving mask. Is otherwise dervied automatically. Defaults to None.
            fixed_img_mask (Optional[numpy.ndarray], optional): Optional fixed mask. Is otherwise dervied automatically. Defaults to None.
            options (Optional[RegistrationOptions], optional): Can be supplied. Otherwise default arguments are used. Defaults to None.

        Returns:
            RegistrationTransforms: _description_
        """
        if options is None:
            options = RegistrationOptions()
        path_temp = options.temporary_directory
        clean_if_exists(path_temp)
        path_temp, _ = derive_subdir(path_temp)
        create_if_not_exists(path_temp)
        path_output = join(path_temp, 'output', 'registrations')
        path_output, subdir_num = derive_subdir(path_output)
        create_if_not_exists(path_output)
        max_size = max(moving_img.shape[0], moving_img.shape[1], fixed_img.shape[0], fixed_img.shape[1])            
        moving_resampling_factor, fixed_resampling_factor = derive_sampling_factors(
            options.pre_sampling_factor,
            max_size,
            options.pre_sampling_max_img_size
        )

        original_moving_image_size = moving_img.shape[:2]
        original_fixed_image_size = fixed_img.shape[:2]
        self.cmdln_returns = []

        reg_params = {
                        'subdir_num': subdir_num
                    }
        moving_preprocessing_params = {
            'resampling_factor': moving_resampling_factor,
            'original_image_size': original_moving_image_size
        }
        fixed_preprocessing_params = {
            'resampling_factor': fixed_resampling_factor,
            'original_image_size': original_fixed_image_size
        }
        paths = {
            'path_temp': path_temp,
            'path_output': path_output,
        }

        # Convert to correct format, if necessary
        moving_img = correct_img_dtype(moving_img)
        fixed_img = correct_img_dtype(fixed_img)
        moving_img = resample_image_sitk(moving_img, moving_resampling_factor)
        fixed_img = resample_image_sitk(fixed_img, fixed_resampling_factor)
        yolo_segmentation_min_size = options.affine_registration_options.preprocessing_options.yolo_segmentation_min_size if options.do_affine_registration else options.nonrigid_registration_options.preprocessing_options.yolo_segmentation_min_size
        if moving_img_mask is None:
            moving_img_mask = self.segmentation_function(moving_img, yolo_segmentation_min_size)
        else:
            moving_img_mask = resample_image_sitk(moving_img_mask, moving_resampling_factor)
        if fixed_img_mask is None:
            fixed_img_mask = self.segmentation_function(fixed_img, yolo_segmentation_min_size)
        else:
            fixed_img_mask = resample_image_sitk(fixed_img_mask, fixed_resampling_factor)
        moving_preprocessing_params['moving_img_mask'] = moving_img_mask
        fixed_preprocessing_params['fixed_img_mask'] = fixed_img_mask
        # Cropping and Padding
        cropped_moving_mask, crop_params_mov = cropping(moving_img_mask)
        cropped_fixed_mask, crop_params_fix = cropping(fixed_img_mask)
        moving_preprocessing_params['cropping_params'] = crop_params_mov
        moving_preprocessing_params['original_shape_image'] = moving_img.shape
        fixed_preprocessing_params['cropping_params'] = crop_params_fix
        fixed_preprocessing_params['original_shape_image'] = fixed_img.shape
        
        cropped_moving_img = moving_img[crop_params_mov[0]:crop_params_mov[1], crop_params_mov[2]:crop_params_mov[3]]
        cropped_fixed_img = fixed_img[crop_params_fix[0]:crop_params_fix[1], crop_params_fix[2]:crop_params_fix[3]]
        moving_pad, fixed_pad = get_symmetric_padding(cropped_moving_img, cropped_fixed_img)
        moving_img = pad_asym(cropped_moving_img, moving_pad)
        fixed_img = pad_asym(cropped_fixed_img, fixed_pad)
        moving_img_mask = pad_asym(cropped_moving_mask, moving_pad)
        fixed_img_mask = pad_asym(cropped_fixed_mask, fixed_pad)
        moving_preprocessing_params['padding'] = moving_pad
        fixed_preprocessing_params['padding'] = fixed_pad
        moving_img = apply_mask(moving_img, moving_img_mask)
        fixed_img = apply_mask(fixed_img, fixed_img_mask)
        _, cropping_padded_mov_mask = cropping(moving_img_mask)
        _, cropping_padded_fix_mask = cropping(fixed_img_mask)
        moving_preprocessing_params['cropped_padded_mask'] = cropping_padded_mov_mask
        fixed_preprocessing_params['cropped_padded_mask'] = cropping_padded_fix_mask

        path_metrics = os.path.join(path_output, 'metrics')
        path_metrics_small_resolution = os.path.join(path_metrics, 'small_resolution')
        path_metrics_full_resolution = os.path.join(path_metrics, 'full_resolution')

        create_if_not_exists(path_metrics)
        create_if_not_exists(path_metrics_small_resolution)
        create_if_not_exists(path_metrics_full_resolution)

        paths['path_metrics'] = path_metrics
        paths['path_metrics_small_resolution'] = path_metrics_small_resolution
        paths['path_metrics_full_resolution'] = path_metrics_full_resolution

        if options.do_affine_registration:
            moving_affine_path = join(path_temp, 'affine_moving_preprocessing')
            fixed_affine_path = join(path_temp, 'affine_fixed_preprocessing')
            moving_img_preprocessed = preprocessing(moving_img,
                                                    options.affine_registration_options.preprocessing_options,
                                                    options.affine_registration_options.resolution,
                                                    options.affine_registration_options.kernel_size,
                                                    moving_affine_path,
                                                    options.affine_registration_options.preprocessing_options.disable_denoising_moving)
            fixed_img_preprocessed = preprocessing(fixed_img,
                                                   options.affine_registration_options.preprocessing_options,
                                                   options.affine_registration_options.resolution,
                                                   options.affine_registration_options.kernel_size,
                                                   fixed_affine_path,
                                                   options.affine_registration_options.preprocessing_options.disable_denoising_fixed)
            height = fixed_img_preprocessed.height
            
            ia_init = ''
            if options.affine_registration_options.ia == 'ia-com-init' and fixed_img_mask is not None and moving_img_mask is not None:
                # Use Segmentation masks to compute center of mass initialization
                # Check that masks are np.arrays
                init_mat_path = os.path.join(path_temp, 'Affine_init.mat')
                init_mat = com_affine_matrix(fixed_img_mask, moving_img_mask)
                write_mat_to_file(init_mat, init_mat_path)
                ia_init = ['-ia', f'{init_mat_path}']
                reg_params['com_x'] = init_mat[0, 2]
                reg_params['com_y'] = init_mat[1, 2]
                # offset = int(translation_length(init_mat[0,2], init_mat[0,1]))
                offset = int((height + (options.affine_registration_options.kernel_size * 4)) / 10)
                n_rigid_iterations = options.affine_registration_options.rigid_iterations
                if n_rigid_iterations == 'auto':
                    n_rigid_iterations = int(1000 * (np.round(np.log10(offset)) + 1))
                    options.affine_registration_options.rigid_iterations = n_rigid_iterations
                    reg_params['rigid_iterations'] = options.affine_registration_options.rigid_iterations
                    reg_params['translation_offset'] = offset
            elif options.affine_registration_options.ia == 'ia-image-centers':
                ia_init = ['-ia-image-centers', '']
                offset = int((height + (options.affine_registration_options.kernel_size * 4)) / 10)
            else:
                print(f'Unknown ia option: {options.affine_registration_options.ia}.')
                offset = int((height + (options.affine_registration_options.kernel_size * 4)) / 10)
            reg_params['offset'] = offset
            reg_params['affine_iteration_vec'] = options.affine_registration_options.iteration_pyramid
            path_small_affine = os.path.join(path_metrics_small_resolution, 'small_affine.mat')
            aff_ret = affine_registration(self.path_to_greedy,
                                            fixed_img_preprocessed.image_path,
                                            moving_img_preprocessed.image_path,
                                            path_small_affine,
                                            offset,
                                            ia_init,
                                            options.affine_registration_options,
                                            self.use_docker_container,
                                            path_temp
                                            )
            self.cmdln_returns.append(aff_ret)    
        else:
            path_small_affine = None  
        paths['path_small_affine'] = path_small_affine

        if options.do_nonrigid_registration:
            (def_reg_ret, 
             new_paths,
             moving_img_preprocessed,
             fixed_img_preprocessed) = self.do_deformable_registration(moving_img,
                                            fixed_img,
                                            moving_img_mask,
                                            fixed_img_mask,
                                            options,
                                            paths,
                                            reg_params)
            self.cmdln_returns.append(def_reg_ret)
            for key in new_paths:
                paths[key] = new_paths[key]

        else:
            paths['path_small_warp'] = None
            paths['path_small_warp_inv'] = None      

        registration_transform = self.reg_postprocess(moving_img_preprocessed,
                             fixed_img_preprocessed,
                             moving_preprocessing_params,
                             fixed_preprocessing_params,
                             options,
                             moving_img,
                             reg_params,
                             paths)
        
        if options.do_nonrigid_registration and options.compute_reverse_nonrigid_registration:
            path_output = join(path_temp, 'output', 'registrations_rev')
            path_output, subdir_num = derive_subdir(path_output)
            create_if_not_exists(path_output)
            paths_rev = paths.copy()
            if 'nonrigid_affine_trans_path' in paths_rev:
                del paths_rev['nonrigid_affine_trans_path']
            del paths_rev['path_small_warp']
            del paths_rev['path_small_warp_inv']
            del paths_rev['path_small_affine']
            paths_rev['path_output'] = path_output
            path_metrics = os.path.join(path_output, 'metrics_rev')
            path_metrics_small_resolution = os.path.join(path_metrics, 'small_resolution')
            path_metrics_full_resolution = os.path.join(path_metrics, 'full_resolution')

            create_if_not_exists(path_metrics)
            create_if_not_exists(path_metrics_small_resolution)
            create_if_not_exists(path_metrics_full_resolution)

            paths_rev['path_metrics'] = path_metrics
            paths_rev['path_metrics_small_resolution'] = path_metrics_small_resolution
            paths_rev['path_metrics_full_resolution'] = path_metrics_full_resolution

            moving_img, fixed_img = fixed_img, moving_img
            moving_img_mask, fixed_img_mask = fixed_img_mask, moving_img_mask
            moving_preprocessing_params, fixed_preprocessing_params = fixed_preprocessing_params, moving_preprocessing_params


            if options.do_affine_registration:
                affine_transform = read_affine_transform(paths['path_small_affine'])
                rev_affine_transform = affine_transform.GetInverse()
                paths_rev['path_small_affine'] = os.path.join(paths_rev['path_metrics_small_resolution'], 'small_affine.mat')
                affine_transform_to_file(rev_affine_transform, paths_rev['path_small_affine'])
            else:
                paths_rev['path_small_affine'] = None
                
            if options.do_nonrigid_registration:
                (def_reg_ret, 
                 new_paths,
                 moving_img_preprocessed,
                 fixed_img_preprocessed) = self.do_deformable_registration(moving_img,
                                                fixed_img,
                                                moving_img_mask,
                                                fixed_img_mask,
                                                options,
                                                paths_rev,
                                                reg_params,
                                                compute_reverse_nonrigid_registration=True)
                self.cmdln_returns.append(def_reg_ret)      
                for key in new_paths:
                    paths_rev[key] = new_paths[key]                  
            rev_registration_transform = self.reg_postprocess(moving_img_preprocessed,
                        fixed_img_preprocessed,
                        moving_preprocessing_params,
                        fixed_preprocessing_params,
                        options,
                        moving_img,
                        reg_params,
                        paths_rev,
                        True)
        else:
            # Overwise just invert the transform.
            rev_registration_transform = RegistrationTransforms(
                forward_transform=registration_transform.backward_transform,
                backward_transform=registration_transform.forward_transform,
            )

        if options.remove_temporary_directory:
            self.__cleanup_temporary_directory(options.temporary_directory)  
        registration_result = RegistrationResult(registration_transform, rev_registration_transform)      
        return registration_result

    def do_deformable_registration(self,
                                   moving_img: numpy.ndarray,
                                   fixed_img: numpy.ndarray,
                                   moving_img_mask: numpy.ndarray,
                                   fixed_img_mask: numpy.ndarray,
                                   options: RegistrationOptions,
                                   paths: dict,
                                   reg_params: dict,
                                   compute_reverse_nonrigid_registration: bool = False) -> tuple[subprocess.CompletedProcess, dict, PreprocessedData, PreprocessedData]:
        """Performs deformable registration.

        Args:
            moving_img (numpy.ndarray): 
            fixed_img (numpy.ndarray): 
            moving_img_mask (numpy.ndarray): 
            fixed_img_mask (numpy.ndarray): 
            options (RegistrationOptions): 
            paths (dict): Paths to various files that have been computed in previous steps.
            reg_params (dict): Collect registration parameters.
            compute_reverse_nonrigid_registration (bool, optional): If True, computes switched moving and fixed image. Defaults to False.

        Returns:
            tuple[subprocess.CompletedProcess, dict, PreprocessedData, PreprocessedData]: Return value from calling greedy, new paths, preprocessed data for moving and fixed image.
        """
        path_metrics_small_resolution = paths['path_metrics_small_resolution']
        if options.do_affine_registration:
            src_resolution = options.affine_registration_options.resolution[0]
            dst_resolution = options.nonrigid_registration_options.resolution[0]

            path_small_affine = paths['path_small_affine']
            if src_resolution == dst_resolution:
                nonrigid_affine_trans_path = path_small_affine
            else:
                resample = src_resolution / dst_resolution * 100
                tmp_factor = 100 / resample    
                affine_transform = rescale_affine(path_small_affine, tmp_factor)
                nonrigid_affine_trans_path = join(path_metrics_small_resolution, 'nr_aff_trans.mat')
                affine_transform_to_file(affine_transform, nonrigid_affine_trans_path)
        else:
            nonrigid_affine_trans_path = None
            resample = None


        path_temp = paths['path_temp']
        reg_params['deformable_iteration_vec'] = options.nonrigid_registration_options.iteration_pyramid
        new_paths = {}
        # Diffeomorphic
        if not compute_reverse_nonrigid_registration:
            moving_nr_path = join(path_temp, 'nr_moving_preprocessing')
            fixed_nr_path = join(path_temp, 'nr_fixed_preprocessing')
            path_small_warp = os.path.join(path_metrics_small_resolution, 'small_warp.nii.gz')
            path_small_warp_inv = os.path.join(path_metrics_small_resolution, 'small_inv_warp.nii.gz')
        else:
            moving_nr_path = join(path_temp, 'nr_moving_preprocessing_reverse')
            fixed_nr_path = join(path_temp, 'nr_fixed_preprocessing_reverse')
            path_small_warp = os.path.join(path_metrics_small_resolution, 'small_warp_reverse.nii.gz')
            path_small_warp_inv = os.path.join(path_metrics_small_resolution, 'small_inv_warp_reverse.nii.gz')
        new_paths['path_small_warp'] = path_small_warp
        new_paths['path_small_warp_inv'] = path_small_warp_inv
        new_paths['nonrigid_affine_trans_path'] = nonrigid_affine_trans_path


        moving_img_preprocessed = preprocessing(moving_img,
                                                options.nonrigid_registration_options.preprocessing_options,
                                                options.nonrigid_registration_options.resolution,
                                                options.nonrigid_registration_options.kernel_size,
                                                moving_nr_path,
                                                options.nonrigid_registration_options.preprocessing_options.disable_denoising_moving)
        fixed_img_preprocessed = preprocessing(fixed_img,
                                            options.nonrigid_registration_options.preprocessing_options,
                                            options.nonrigid_registration_options.resolution,
                                            options.nonrigid_registration_options.kernel_size,
                                            fixed_nr_path,
                                            options.nonrigid_registration_options.preprocessing_options.disable_denoising_fixed)
        height = fixed_img_preprocessed.height

        ia_init = ''
        if options.nonrigid_registration_options.ia == 'ia-com-init' and fixed_img_mask is not None and moving_img_mask is not None:
            # Use Segmentation masks to compute center of mass initialization
            # Check that masks are np.arrays
            init_mat_path = os.path.join(path_temp, 'Affine_init.mat')
            init_mat = com_affine_matrix(fixed_img_mask, moving_img_mask)
            write_mat_to_file(init_mat, init_mat_path)
            ia_init = ['-ia', f'{init_mat_path}']
            reg_params['com_x'] = init_mat[0, 2]
            reg_params['com_y'] = init_mat[1, 2]
        elif options.nonrigid_registration_options.ia == 'ia-image-centers':
            ia_init = ['-ia-image-centers', '']
        else:
            print(f'Unknown ia option: {options.nonrigid_registration_options.ia}.')

        deformable_reg_ret = deformable_registration(self.path_to_greedy,
                                                        fixed_img_preprocessed.image_path,
                                                        moving_img_preprocessed.image_path,
                                                        options.nonrigid_registration_options,
                                                        output_warp=path_small_warp,
                                                        output_inv_warp=path_small_warp_inv,
                                                        affine_pre_transform=nonrigid_affine_trans_path,
                                                        ia=ia_init,
                                                        use_docker_container=self.use_docker_container,
                                                        temp_directory=path_temp)
        return deformable_reg_ret, new_paths, moving_img_preprocessed, fixed_img_preprocessed
           
    def reg_postprocess(self,
                        moving_img_preprocessed: PreprocessedData,
                        fixed_img_preprocessed: PreprocessedData,
                        moving_preprocessing_params: dict,
                        fixed_preprocessing_params: dict,
                        options: RegistrationOptions,
                        moving_img: numpy.ndarray,
                        reg_params: dict,
                        paths: dict,
                        reverse: bool = False) -> RegistrationResult:
        """Post processing of registration. Rescales and/or composes affine and
        nonrigid registration.

        Args:
            moving_img_preprocessed (PreprocessedData): 
            fixed_img_preprocessed (PreprocessedData): 
            moving_preprocessing_params (dict): 
            fixed_preprocessing_params (dict): 
            options (RegistrationOptions): 
            moving_img (numpy.ndarray): 
            reg_params (dict): 
            paths (dict): 
            reverse (bool, optional): Defaults to False.

        Returns:
            RegistrationResult: 
        """
        path_output = paths['path_output']
        path_small_affine = paths['path_small_affine']

        original_moving_image_size = moving_preprocessing_params['original_image_size']
        original_fixed_image_size = fixed_preprocessing_params['original_image_size']
        # Write small ref image to file for warping of coordinates
        small_fixed = sitk.ReadImage(fixed_img_preprocessed.image_path)
        empty_fixed_img = small_fixed[:,:]
        empty_fixed_img[:,:] = 0
        path_to_small_ref_image = join(path_output, 'small_ref_image.nii.gz')
        sitk.WriteImage(empty_fixed_img, path_to_small_ref_image)

        # If no non-rigid registration is performed we keep the affine transform unbounded.
        # TODO: This needs some changing. There should be an option not to composite affine and nonrigid registrations, so that affine keeps being unbounded.
        if options.do_nonrigid_registration:
            path_metrics_small_resolution = paths['path_metrics_small_resolution']
            path_metrics_full_resolution = paths['path_metrics_full_resolution']
            path_small_warp = paths['path_small_warp']
            path_small_warp_inv = paths['path_small_warp_inv']
            no_2_orig_resample = (options.nonrigid_registration_options.resolution[0] / moving_img.shape[0]) * 100
            no_2_orig_factor = 100 / no_2_orig_resample            
            if options.do_affine_registration and not options.affine_registration_options.keep_affine_transform_unbounded:
                path_small_composite_warp = os.path.join(path_metrics_small_resolution, 'small_composite_warp.nii.gz')
                nonrigid_affine_trans_path = paths['nonrigid_affine_trans_path']
                composite_warps(
                    self.path_to_greedy,
                    nonrigid_affine_trans_path,
                    path_small_warp,
                    path_to_small_ref_image,
                    path_small_composite_warp            
                )
                path_big_composite_warp = os.path.join(path_metrics_full_resolution, 'big_composite_warp.nii.gz')
                rescale_warp(
                    path_small_composite_warp,
                    path_big_composite_warp,
                    (fixed_img_preprocessed.width, fixed_img_preprocessed.height),
                    (fixed_img_preprocessed.width_original,
                        fixed_img_preprocessed.height_original),
                    no_2_orig_factor)
                path_small_inverted_composite_warp = os.path.join(path_metrics_small_resolution, 'small_inv_composite_warp.nii.gz')
                composite_warps(
                    self.path_to_greedy,
                    nonrigid_affine_trans_path,
                    path_small_warp_inv,
                    path_to_small_ref_image,
                    path_small_inverted_composite_warp,
                    invert=True 
                )
                path_big_composite_warp_inv = os.path.join(path_metrics_full_resolution, 'big_inv_composite_warp.nii.gz')
                rescale_warp(
                    path_small_inverted_composite_warp,
                    path_big_composite_warp_inv,
                    (moving_img_preprocessed.width, moving_img_preprocessed.height),
                    (moving_img_preprocessed.width_original,
                        moving_img_preprocessed.height_original),
                    no_2_orig_factor)

                forward_deformable_transform = realign_displacement_field(path_big_warp)
                backward_deformable_transform = realign_displacement_field(path_big_warp_inv)
            elif options.do_affine_registration and options.affine_registration_options.keep_affine_transform_unbounded:
                # First rescale affine transforms
                aff_2_orig_resample = options.affine_registration_options.resolution[0] / moving_img.shape[0] * 100
                aff_2_orig_factor = 100 / aff_2_orig_resample
                forward_affine_transform = rescale_affine(path_small_affine, aff_2_orig_factor)
                backward_affine_transform = forward_affine_transform.GetInverse()
                path_big_warp = os.path.join(path_metrics_full_resolution, 'big_warp.nii.gz')
                rescale_warp(
                    path_small_warp,
                    path_big_warp,
                    (fixed_img_preprocessed.width, fixed_img_preprocessed.height),
                    (fixed_img_preprocessed.width_original,
                        fixed_img_preprocessed.height_original),
                    no_2_orig_factor)
                
                path_big_warp_inv = os.path.join(path_metrics_full_resolution, 'big_inv_warp.nii.gz')
                rescale_warp(
                    path_small_warp_inv,
                    path_big_warp_inv,
                    (fixed_img_preprocessed.width, fixed_img_preprocessed.height),
                    (fixed_img_preprocessed.width_original,
                        fixed_img_preprocessed.height_original),
                    no_2_orig_factor)

                forward_deformable_transform = realign_displacement_field(path_big_warp)
                backward_deformable_transform = realign_displacement_field(path_big_warp_inv)
                    
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

            elif not options.do_affine_registration:
                # First rescale affine transforms       
                path_big_warp = os.path.join(path_metrics_full_resolution, 'big_warp.nii.gz')
                rescale_warp(
                    path_small_warp,
                    path_big_warp,
                    (fixed_img_preprocessed.width, fixed_img_preprocessed.height),
                    (fixed_img_preprocessed.width_original,
                        fixed_img_preprocessed.height_original),
                    no_2_orig_factor)
                
                path_big_warp_inv = os.path.join(path_metrics_full_resolution, 'big_inv_warp.nii.gz')
                rescale_warp(
                    path_small_warp_inv,
                    path_big_warp_inv,
                    (fixed_img_preprocessed.width, fixed_img_preprocessed.height),
                    (fixed_img_preprocessed.width_original,
                        fixed_img_preprocessed.height_original),
                    no_2_orig_factor)

                forward_deformable_transform = realign_displacement_field(path_big_warp)
                backward_deformable_transform = realign_displacement_field(path_big_warp_inv)
                    
                forward_transform = sitk.CompositeTransform(2)
                forward_transform.AddTransform(forward_deformable_transform)
                
                backward_transform = sitk.CompositeTransform(2)
                backward_transform.AddTransform(backward_deformable_transform)

                path_small_composite_warp = ''
                path_small_inverted_composite_warp = ''
                path_big_composite_warp = ''
                path_big_composite_warp_inv = ''

        else:
            # This case is triggered when no nonrigid registration is computed.
            aff_2_orig_resample = options.affine_registration_options.resolution[0] / moving_img.shape[0] * 100
            aff_2_orig_factor = 100 / aff_2_orig_resample            
            # Set some paths to empty. Will remove them entirely later.
            path_small_composite_warp = ''
            path_small_inverted_composite_warp = ''
            path_big_composite_warp = ''
            path_big_composite_warp_inv = ''
            # TODO: Do I need to -1 this transformation? (Because the direction was different for the displacement field. Though things seem to work with
            # affine transformation.) Check!!
            forward_transform = rescale_affine(path_small_affine, aff_2_orig_factor)
            backward_transform = forward_transform.GetInverse()


        displacement_field = None
        inv_displacement_field = None
        reg_result = InternalRegParams(
            path_to_small_moving=moving_img_preprocessed.image_path,
            path_to_small_fixed=fixed_img_preprocessed.image_path,
            path_to_small_composite=path_small_composite_warp,
            path_to_big_composite=path_big_composite_warp,
            path_to_small_inv_composite=path_small_inverted_composite_warp,
            path_to_big_inv_composite=path_big_composite_warp_inv,
            cmdl_log=self.cmdln_returns,
            reg_params=reg_params,
            moving_preprocessing_params=moving_preprocessing_params,
            fixed_preprocessing_params=fixed_preprocessing_params,
            path_to_small_ref_image=path_to_small_ref_image,
            sub_dir_key=reg_params['subdir_num'],
            displacement_field=displacement_field,
            inv_displacement_field=inv_displacement_field
        )

        composited_forward_transform = compose_reg_transforms(forward_transform, 
                                                              moving_preprocessing_params,
                                                              fixed_preprocessing_params)
        composited_backward_transform = compose_inv_reg_transforms(backward_transform, 
                                                                   moving_preprocessing_params,
                                                                   fixed_preprocessing_params)
        fixed_transform = GFHTransform(original_fixed_image_size, composited_forward_transform)
        moving_transform = GFHTransform(original_moving_image_size, composited_backward_transform)
        registration_result = RegistrationTransforms(forward_transform=fixed_transform, backward_transform=moving_transform, cmdln_returns=self.cmdln_returns, reg_params=reg_result)
        # Return this!
        return registration_result


    # TODO: Add option for skipping affine registration.
    def groupwise_registration(self,
                               image_mask_list: list[tuple[numpy.ndarray, numpy.ndarray | None]],
                               options: RegistrationOptions | None = None,
                               ) -> tuple[GroupwiseRegResult, list[numpy.ndarray]]:
        """Performs groupwise registration on a provided image list. 
        For each image, an optional mask can be provided. Fixed 
        image is last image in image_mask_list.

        Every other image is a moving image. Groupwise registration 
        is performed in 2 steps:
            1. Pairwise affine registration between all adjacent images
               is computed and applied such that each moving image is 
               affinely registered onto the fixed image.
            2. Nonrigid registration between each moving image and 
               the fixed image.


        Args:
            image_mask_list (List[Tuple[numpy.ndarray, Optional[numpy.ndarray]]]): List of images. Last image is fixed image. Every other image is a moving image. For each image, 
            an optional mask can be supplied.
            options (Optional[RegistrationOptions], optional): Registration options. At this moment, the affine registration is
            always executed. `options.do_affine_registration` is ignored, but the nonrigid registration can be disabled.
            Defaults to None.

        Returns:
            Tuple[GroupwiseRegResult, List[numpy.ndarray]]: GroupwiseRegResult contains all computed transformations. List of images are either affine or nonrigid warped images.
        """
        if options is None:
            options = RegistrationOptions()
        do_nr_registration = options.do_nonrigid_registration
        # Stage1: Affine register along the the sequence.
        moving_tuple = image_mask_list[0]
        if isinstance(moving_tuple, tuple):
            moving_image, moving_mask = moving_tuple
        else:
            moving_image = moving_tuple
            moving_mask = None
        affine_transform_lists = []
        # We set this to false for the affine part and set it back to the original value afterwards.
        options.do_nonrigid_registration = False
        options.do_affine_registration = True
        reverse_affine_transform_list = []
        for fixed_tuple in image_mask_list[1:]:
            if isinstance(fixed_tuple, tuple):
                fixed_image, fixed_mask = fixed_tuple
            else:
                fixed_image = fixed_tuple
                fixed_mask = None
            reg_result = self.register_(moving_image, fixed_image, moving_mask, fixed_mask, options)
            affine_transform_lists.append(reg_result.registration)
            reverse_affine_transform_list.append(reg_result.reverse_registration)
            moving_image = fixed_image
            moving_mask = fixed_mask
        # Stage 2: Take the matched images and do a nonrigid registration
        if not do_nr_registration:
            g_res = GroupwiseRegResult(affine_transform_lists, reverse_affine_transform_list, [], [])
            return g_res, None
        options.do_affine_registration = False
        options.do_nonrigid_registration = True
        nonrigid_transformations = []
        reverse_nonrigid_transformations = []
        nonrigid_warped_images = []
        fixed_tuple = image_mask_list[-1]
        if isinstance(fixed_tuple, tuple):
            fixed_image, fixed_mask = fixed_tuple
        else:
            fixed_image = fixed_tuple
            fixed_mask = None
        for idx, moving_tuple in enumerate(image_mask_list[:-1]):
            if isinstance(moving_tuple, tuple):
                moving_image, moving_mask = moving_tuple
            else:
                moving_image = moving_tuple
                moving_mask = None
            if moving_mask is None:
                moving_mask = self.segmentation_function(moving_image)
            composited_fixed_transform = compose_transforms([x.forward_transform for x in affine_transform_lists][idx:])
            warped_image = self.transform_image(moving_image, composited_fixed_transform, 'LINEAR')
            warped_mask = self.transform_image(moving_mask, composited_fixed_transform, 'NN')
            nonrigid_reg_result = self.register(warped_image, fixed_image, warped_mask, fixed_mask, options=options)
            deformable_warped_image = self.transform_image(warped_image, nonrigid_reg_result.registration.forward_transform, 'LINEAR')
            nonrigid_warped_images.append(deformable_warped_image)
            nonrigid_transformations.append(nonrigid_reg_result.registration)
            reverse_nonrigid_transformations.append(nonrigid_reg_result.reverse_registration)
        reverse_nonrigid_transformations = reverse_nonrigid_transformations[::-1]
        groupwise_registration_results = GroupwiseRegResult(affine_transform_lists, 
                                                            reverse_affine_transform_list,
                                                            nonrigid_transformations,
                                                            reverse_nonrigid_transformations)
        return groupwise_registration_results, nonrigid_warped_images

    def transform_image(self,
                        image: numpy.ndarray,
                        transform: 'GFHTransform',
                        interpolation_mode: str = 'LINEAR') -> numpy.ndarray:
        """Transforms image data from moving to fixed image space using computed transformation. Use forward_transform attribute.

        Args:
            image (numpy.ndarray): 
            transform (GFHTransform): 
            interpolation_mode (str, optional): Defaults to 'LINEAR'.

        Returns:
            numpy.ndarray: 
        """
        return self.transform_image_(image,
                                     transform.transform,
                                     transform.size,
                                     interpolation_mode)

    def transform_image_(self,
                        image: numpy.ndarray, 
                        transform: SimpleITK.SimpleITK.Transform,
                        size: tuple[int,int],
                        interpolation_mode: str = 'LINEAR') -> numpy.ndarray:
        """Transforms image from moving to fixed image space.

        Args:
            image (numpy.ndarray): 
            transform (SimpleITK.SimpleITK.Transform): 
            size (Tuple[int,int]): Fixed image space resolution.
            interpolation_mode (str, optional): 'LINEAR' or 'NN'. Defaults to 'LINEAR'.

        Returns:
            numpy.ndarray: _description_
        """
        interpolator = sitk.sitkLinear if interpolation_mode == 'LINEAR' else sitk.sitkNearestNeighbor
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
                           pointset: numpy.ndarray,
                           transform: GFHTransform) -> numpy.ndarray:
        """Transforms pointset from moving to fixed image space. Use backward_transform attribute.

        Args:
            pointset (numpy.ndarray): 
            transform (GFHTransform): 

        Returns:
            numpy.ndarray:
        """
        return self.transform_pointset_(
            pointset,
            transform.transform
        )

    def transform_pointset_(self,
                         pointset: numpy.ndarray,
                         transform: SimpleITK.SimpleITK.Transform) -> numpy.ndarray:
        """Transform pointset from moving to fixed image space.
    
        Args:
            pointset (numpy.ndarray): 
            transformation (SimpleITK.SimpleITK.Transform): 
    
        Returns:
            numpy.ndarray: 
        """
        pointset -= 0.5
        warped_points = []
        for i in range(pointset.shape[0]):
            point = (pointset[i,0], pointset[i,1])
            warped_point = transform.TransformPoint(point)
            warped_points.append(warped_point)
        warped_pointset = np.array(warped_points)
        return warped_pointset

    # TODO: Fix types for geojson.
    def transform_geojson(self,
                          geojson_data: geojson.GeoJSON,
                          transformation: SimpleITK.SimpleITK.Transform,
                          **kwards) -> list[geojson.GeoJSON] | geojson.GeoJSON:
        """Applies transformation to geojson data. Can be a feature collection ot a list of features.

        Args:
            geojson_data (geojson.GeoJSON): 
            transformation (SimpleITK.SimpleITK.Image): 

        Returns:
            list[geojson.GeoJSON] | geojson.GeoJSON: 
        """
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

    def __warp_geojson_coord_tuple(self, coord: tuple[float, float], transform: SimpleITK.SimpleITK.Transform) -> tuple[float, float]:
        """Transforms coordinates from geojson data from moving to fixed image space.

        Args:
            coord (Tuple[float, float]): 
            transform (SimpleITK.SimpleITK.Transform): 

        Returns:
            Tuple[float, float]: 
        """
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
    def load_from_config(cls, config: dict[str, Any] | None = None) -> 'GreedyFHist':
        """Loads GreedyFHist registerer using additional arguments supplied in config.

        Args:
            config (Dict[str, Any]): _description_

        Returns:
            GreedyFHist: _description_
        """
        # Refers to greedy's directory. If not supplied, assumes that greedy is in PATH.
        if config is None:
            config = {}
        path_to_greedy = config.get('path_to_greedy', '')
        path_to_greedy = join(path_to_greedy, 'greedy')
        seg_fun = load_yolo_segmentation()
        return cls(path_to_greedy=path_to_greedy, segmentation_function=seg_fun)