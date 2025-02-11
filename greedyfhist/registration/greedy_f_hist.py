"""GreedyFHist, the registration algorithm, includes pairwise/groupwise registration, transform for various file formats. 

This module handles the core registration functionality via the 
GreedyFHist class. Results are exported as GroupwiseRegResult or 
RegistrationTransforms. 
"""

from copy import deepcopy
from dataclasses import dataclass
import multiprocess
import os
from os.path import join
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Callable

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
    read_affine_transform,
    get_corner_pixels
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

from greedyfhist.segmentation import load_segmentation_function, load_yolo_segmentation

from greedyfhist.options import (
    RegistrationOptions, 
    PreprocessingOptions, 
    SegmentationOptions
)

from greedyfhist.utils.tiling import (
    reassemble_sitk_displacement_field, 
    ImageTile, 
    extract_image_tiles, 
    get_tile_params,
    get_tile_params_by_tile_size,
    extract_image_tiles_from_tile_sizes,
    reassemble_sitk_displacement_field_from_tile_size
)


def _preprocessing(image: numpy.ndarray,
                  preprocessing_options: PreprocessingOptions,
                  resolution: tuple[int, int],
                  kernel_size: int,
                  tmp_path: str,
                  skip_denoising: bool = False,
                  padding_mode: str = 'mask'
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
        padding_mode (str | None): Either 'mask' or 'nomask'. If 'mask' is used, it is assumed that background
            segmentation took place and the value padding is 0. If 'nomask' is used, it is assumed that no background
            segmentation has been performed and the padding strategy is the same for HistoReg: From the intensity values
            of the image corners a gaussian distribution is estimated and the padding values are sampled from that gaussian
            distribution.


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
    image_preprocessed = _downsample_and_padding(image,
                                                     kernel_size,
                                                     resolution,
                                                     smoothing,
                                                     tmp_dir,
                                                     padding_mode)
    return image_preprocessed


def _downsample_and_padding(image: numpy.ndarray,
                     kernel: int,
                     resolution: tuple[int, int],
                     smoothing: int,
                     tmp_dir: str,
                     padding_for_mask: bool = True) -> 'PreprocessedData':
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
        padding_for_mask (bool): If True is used, the
            padding value is 0. If False is used, it is assumed that no
            background segmentation took place. In that case we use HistoReg's approach and 
            pad the image with a gaussian distribution.

    Returns:
        PreprocessedData: Contains path to downscaled image and 
                          additional image parameters.
    """
    small_image = resample_image_with_gaussian(image, resolution, smoothing)
    height_image = image.shape[1]
    width_image = image.shape[0]

    height_small_image = small_image.shape[1]
    width_small_image = small_image.shape[0]

    img_padded = pad_image(small_image, kernel * 4)
    if not padding_for_mask:
        corner_pixels = get_corner_pixels(image, kernel, kernel)
        mean_c_ints = corner_pixels.mean()
        var_c_ints = corner_pixels.var()
        img_outline = np.ones_like(image)
        padded_outline = pad_image(img_outline, kernel * 4)
        padded_outline_inv = np.abs(padded_outline.astype(int) - 1).astype(np.uint8)
        noisy_data = np.random.normal(mean_c_ints, var_c_ints, padded_outline_inv.shape)
        noised_padding = padded_outline_inv * noisy_data
        img_padded = img_padded + noised_padding

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


def _reassemble_to_gfh_transform_from_tile_size(displ_tiles: list[ImageTile], 
                                trx_shape: tuple[int, int] | tuple[int, int, int]) -> GFHTransform:
    """Reassembles transformations for each tile into one transformation.

    Args:
        displ_tiles (list[ImageTile]): 
        trx_shape (tuple[int, int] | tuple[int, int, int]): 

    Returns:
        GFHTransform: 
    """
    # displ_np = reassemble_sitk_displacement_field(displ_tiles, trx_shape)
    displ_np = reassemble_sitk_displacement_field_from_tile_size(displ_tiles, trx_shape)
    displ_sitk = sitk.GetImageFromArray(displ_np, True)
    displ_sitk = sitk.Cast(displ_sitk, sitk.sitkVectorFloat64)
    trx = sitk.DisplacementFieldTransform(displ_sitk)
    gfh_transform = GFHTransform(trx_shape, trx)
    return gfh_transform


def _reassemble_to_gfh_transform(displ_tiles: list[ImageTile], 
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


# TODO: Set paths
def register_from_filepaths(moving_img_path: str,
                            fixed_img_path: str,
                            target_img_path: str,
                            moving_img_mask_path: str | None = None,
                            fixed_img_mask_path: str | None = None,
                            options: RegistrationOptions | None = None,
                            transform_path: str | None = False,
                            ) -> tuple['RegistrationResult', numpy.ndarray | None]:
    """Registers two iamges based on their filepath. 
    
    Args:
        moving_img_path (str): Path to moving image.
        fixed_img_path (str): Path to fixed image.
        target_img_path (str): Path to target image.
        moving_img_mask_path (str|None): Path to moving image mask. Defaults to None.
        fixed_img_mask_path (str|None): Path to fixed image mask. Defaults to None.
        options (RegistrationOptions|None): Contains all registration options. If None, uses default options.
        transform_path (str|None): Path to strogin transformation. Defaults to None.

    Returns:
        tuple[RegistrationResult, numpy.ndarray | None]: Computed transformation and transformed image.
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
    registration_result = register(moving_image,
                                        fixed_image,
                                        moving_img_mask,
                                        fixed_img_mask,
                                        options)
    
    warped_image = transform_image(moving_image, registration_result.registration.forward_transform, sitk.sitkBSpline2)
    write_to_ometiffile(warped_image,
                        target_img_path,
                        metadata=fix_metadata)
    if transform_path is not None:
        registration_result.to_directory(transform_path)
    return registration_result, warped_image


def groupwise_registration_from_filepaths(
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
    if options is None:
        options = RegistrationOptions()
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
    groupwise_registration_result, warped_images = groupwise_registration(img_mask_list, options)
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


def register_tiles(moving_image_tiles: list[ImageTile],
                   fixed_image_tiles: list[ImageTile],
                   tile_reg_opts: RegistrationOptions | None = None,
                   verbose: bool = False) -> RegistrationResult:
    """Register moving and fixed image tiles. Each tile in `moving_image_tiles` is registered
    with a corresponding tile at the same position as `fixed_image_tiles`. 

    Args:
        moving_image_tiles (list[ImageTile]): List of moving image tiles.
        fixed_image_tiles (list[ImageTile]): List of fixed image tiles.
        tile_reg_opts (RegistrationOptions | None, optional): Nonrigid registration options applied to register pairs
        of tiles. Defaults to None.
        verbose (bool, optional): Defaults to False.

    Returns:
        RegistrationResult: Computed transformation
    """
    fw_displ_tiles = []
    bw_displ_tiles = []
    if verbose:
        print('Starting registration of tiles.')
        print('tile_reg_opts: ', tile_reg_opts)
    for idx, (moving_image_tile_, fixed_image_tile_) in tqdm.tqdm(enumerate(zip(moving_image_tiles, fixed_image_tiles)), disable=not verbose):
        if tile_reg_opts is None:
            if verbose:
                print('No options for registering tiles supplied. Using default.')
            nrrt_options = RegistrationOptions.nrpt_only_options()
        else:
            nrrt_options = deepcopy(tile_reg_opts)
            
        if verbose:
            print(nrrt_options)

        moving_image_ = moving_image_tile_.image
        fixed_image_ = fixed_image_tile_.image
        mask = np.ones(moving_image_.shape[:2], dtype=np.uint8)

        try:
            if verbose:
                start = time.time()
            # Call internatl `register_` to avoid wrapping context.
            registration_result_ = _perform_registration(moving_img=moving_image_,
                                                    fixed_img=fixed_image_,
                                                    moving_img_mask=mask,
                                                    fixed_img_mask=mask,
                                                    options=nrrt_options)
            if verbose:
                end = time.time()
                print(f'Duration of register function: {end - start}')
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
    fixed_transform = _reassemble_to_gfh_transform(fw_displ_tiles, fixed_image_tiles[0].original_shape)
    moving_transform = _reassemble_to_gfh_transform(bw_displ_tiles, moving_image_tiles[0].original_shape)
    registration_transforms = RegistrationTransforms(forward_transform=fixed_transform, backward_transform=moving_transform)
    rev_registration_transforms = RegistrationTransforms(forward_transform=moving_transform, backward_transform=fixed_transform)
    reg_result = RegistrationResult(registration_transforms, rev_registration_transforms)
    if verbose:
        reg_result.reg_params = [fw_displ_tiles, bw_displ_tile]
    return RegistrationResult(registration_transforms, rev_registration_transforms)  


def _register_tiles_from_tile_size(
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
    if tile_reg_opts is None:
        tile_reg_opts = RegistrationOptions()
    fw_displ_tiles = []
    bw_displ_tiles = []
    if tile_reg_opts.tiling_options.n_procs is None:
        for idx, (moving_image_tile_, fixed_image_tile_) in tqdm.tqdm(enumerate(zip(moving_image_tiles, fixed_image_tiles)), 
                                                                    disable=not verbose,
                                                                    total=len(moving_image_tiles)):
            fw_displ_tile, bw_displ_tile = _register_tiles_helper(moving_image_tile_,
                                                                    fixed_image_tile_,
                                                                    tile_reg_opts,
                                                                    verbose)
            fw_displ_tiles.append(fw_displ_tile)
            bw_displ_tiles.append(bw_displ_tile)
    else:
        if verbose:
            print(f'Number of pools used: {tile_reg_opts.tiling_options.n_procs}')
        tile_reg_opts.segmentation = None
        tile_reg_opts.disable_mask_generation = True
        mp_args = []
        for i in range(len(moving_image_tiles)):
            moving_image_tile = moving_image_tiles[i]
            fixed_image_tile = fixed_image_tiles[i]
            tile_reg_opts_ = deepcopy(tile_reg_opts)
            tile_reg_opts_.temporary_directory = f'{tile_reg_opts_.temporary_directory}/tiling_registration/{i}'
            tile_reg_opts_.segmentation = None
            tile_reg_opts_.disable_mask_generation = True
            mp_args.append((moving_image_tile, fixed_image_tile, tile_reg_opts_, verbose))
        pool = multiprocess.Pool(tile_reg_opts.tiling_options.n_procs)
        displ_tiles = pool.starmap(_register_tiles_helper, mp_args)
        for (fw_displ_tile, bw_displ_tile) in displ_tiles:
            fw_displ_tiles.append(fw_displ_tile)
            bw_displ_tiles.append(bw_displ_tile)
    fixed_transform = _reassemble_to_gfh_transform_from_tile_size(fw_displ_tiles, fixed_image_tiles[0].original_shape)
    moving_transform = _reassemble_to_gfh_transform_from_tile_size(bw_displ_tiles, moving_image_tiles[0].original_shape)
    registration_transforms = RegistrationTransforms(forward_transform=fixed_transform, backward_transform=moving_transform)
    rev_registration_transforms = RegistrationTransforms(forward_transform=moving_transform, backward_transform=fixed_transform)
    reg_result = RegistrationResult(registration_transforms, rev_registration_transforms)
    if verbose:
        reg_result.registration.reg_params = [fw_displ_tiles, bw_displ_tile]
    return RegistrationResult(registration_transforms, rev_registration_transforms)  


def _register_tiles_helper(moving_image_tile: ImageTile,
                          fixed_image_tile: ImageTile,
                          tiling_options: RegistrationOptions,
                          verbose: bool = False):
    """
    Helper function for executing tiling registration concurrently.
    
    Args:
        moving_image_tile (ImageTile):
            Image tiles of moving image.
        fixed_image_tile (ImageTile):
            Image tiles of fixed image.
        tile_reg_opts (RegistrationOptions):
            Options for registration of tiles.
        verbose (bool) = False
    
    Returns: (ImageTile, ImageTile)
        Forward and backward transformation in tile context.
    """
    moving_image_ = moving_image_tile.image
    fixed_image_ = fixed_image_tile.image
    mask = np.ones(moving_image_.shape[:2], dtype=np.uint8)

    try:
        # Call interal `_perform_registration` to avoid wrapping context.
        registration_result_ = _perform_registration(moving_img=moving_image_,
                                                fixed_img=fixed_image_,
                                                moving_img_mask=mask,
                                                fixed_img_mask=mask,
                                                options=tiling_options)
        transform_to_displacementfield = sitk.TransformToDisplacementField(
            registration_result_.registration.forward_transform.transform,
            outputPixelType=sitk.sitkVectorFloat64,
            size=registration_result_.registration.forward_transform.size[::-1]
        )
        fw_displ_tile = ImageTile(transform_to_displacementfield,
                            fixed_image_tile.x_props,
                            fixed_image_tile.y_props,
                            
        )
        # Backward transform
        transform_to_displacementfield = sitk.TransformToDisplacementField(
            registration_result_.registration.backward_transform.transform,
            outputPixelType=sitk.sitkVectorFloat64,
            size=registration_result_.registration.backward_transform.size[::-1]
        )
        bw_displ_tile = ImageTile(transform_to_displacementfield,
                            fixed_image_tile.x_props,
                            fixed_image_tile.y_props
        )        
    except Exception as e:
        if verbose:
            print(f'Something went wrong in tile.')
            print(e)
        # TODO: Fix this! Needs to be a transform.
        fw_tile = sitk.GetImageFromArray(np.dstack((mask.astype(np.float64), mask.astype(np.float64))), True)
        fw_tile = sitk.Cast(fw_tile, sitk.sitkVectorFloat64)
        fw_displ_tile = ImageTile(fw_tile,
                            fixed_image_tile.x_props,
                            fixed_image_tile.y_props)
        bw_tile = sitk.GetImageFromArray(np.dstack((mask.astype(np.float64), mask.astype(np.float64))), True)
        bw_tile = sitk.Cast(bw_tile, sitk.sitkVectorFloat64)
        bw_displ_tile = ImageTile(bw_tile,
                            moving_image_tile.x_props,
                            moving_image_tile.y_props)
    return fw_displ_tile, bw_displ_tile


def simple_tiling_registration(moving_image: numpy.ndarray,
                        fixed_image: numpy.ndarray,
                        registration_options: RegistrationOptions,
                        verbose: bool = False) -> RegistrationResult:
    """Simple tiling based registration. (Also sometimes referred to as patch-based). Tiling registration is an alternative to the standard
    non-rigid registration. The advantage of tiling is that removal of features due to downsizing can be avoided at an increased runtime 
    cost. 
    
    In simple tiling moving and fixed image are separated into tiles with some user-defined minium overlap. After each pair of tiles
    has been registered, the tiled transformations are joined into one transform.
    
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
    return _simple_tiling_registration(
        moving_image,
        fixed_image,
        tile_options=registration_options,
        verbose=verbose
    )


def _simple_tiling_registration(
                    moving_image: numpy.ndarray,
                    fixed_image: numpy.ndarray,
                    tile_options: RegistrationOptions | None = None,
                    verbose: bool = False) -> RegistrationResult:
    """Tiling based registration. (Also sometimes referred to as patch-based). Tiling registration is an alternative to the standard
    non-rigid registration. The advantage of tiling is that removal of features due to downsizing can be avoided at an increased runtime 
    cost. Tiling comes in two modes: simple and pyramidical. 
    
    In simple tiling moving and fixed image are separated into tiles with some user-defined minium overlap. After each pair of tiles
    has been registered, the tiled transformations are joined into one transform.
    
    In pyramid tiling, moving and fixed image on each level are separated into tiles, registered, and transformation matrices are put 
    back together. On each new level, the tile size is reduced until a stop criteria is reached.
    
    Note: Images are expected to be of the same shape. Performing an affine or nonrigid registration beforehand guarantees that both
    input images have the same shape.

    Args:
        moving_image (numpy.ndarray): Source image.
        fixed_image (numpy.ndarray): Target image.
        tile_size: 
        verbose (bool, optional): Prints more information. Defaults to False.

    Returns:
        RegistrationResult: 
    """
    
    if tile_options is None: 
        tile_options = RegistrationOptions()
    # We set this to False skip affine registration when we call `_perform_registration`
    tile_options.do_affine_registration = False
    
    tile_size = tile_options.tiling_options.tile_size
    if isinstance(tile_size, int):
        tile_size = (tile_size, tile_size)

    # Should it start with 0 or not. 
    x_tile_props = get_tile_params_by_tile_size(moving_image.shape[0], 
                                                tile_size=tile_size[0], 
                                                min_overlap=tile_options.tiling_options.min_overlap)
    y_tile_props = get_tile_params_by_tile_size(moving_image.shape[1], 
                                                tile_size=tile_size[1], 
                                                min_overlap=tile_options.tiling_options.min_overlap)
    
    moving_image_tiles = extract_image_tiles_from_tile_sizes(moving_image, x_tile_props, y_tile_props)
    fixed_image_tiles = extract_image_tiles_from_tile_sizes(fixed_image, x_tile_props, y_tile_props)
    
    if verbose:
        print(f'Number of extracted tiles: {len(moving_image_tiles)}.')
        
    reg_result = _register_tiles_from_tile_size(moving_image_tiles, 
                                                    fixed_image_tiles, 
                                                    tile_options, 
                                                    verbose)
    return reg_result


# TODO: Add concurrency
def pyramid_tiling_registration(moving_image: numpy.ndarray,
                      fixed_image: numpy.ndarray,
                      registration_options: RegistrationOptions,
                      verbose: bool = False) -> RegistrationResult:
    """Pyramidical tiling nonrigid registration (ptnr) mode. A registration modus that can be used in addition, or instead
    of nonrigid registration. 
    
    The base idea behind nrpt is that images are divided into tiles which can then be registered one-by-one due to the locality of 
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
    nrpt_opts = registration_options.tiling_options
    return _pyramid_tiling_registration(
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


def _pyramid_tiling_registration(
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
    
    The base idea behind nrpt is that images are divided into tiles which can then be registered one-by-one due to the locality of 
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
        RegistrationResult: 
    """
    
    if nrpt_tile_options is None: 
        nrpt_tile_options = RegistrationOptions()
    # We set this to False skip affine registration when we call `_perform_registration`
    nrpt_tile_options.do_affine_registration = False
    
    temp_image = moving_image.copy()

    # Should it start with 0 or not. 
    loop_counter = 0

    reg_results = []
    temp_images = [temp_image]

    actual_tile_size = np.inf

    downsampling_size = nrpt_tile_options.nonrigid_registration_options.resolution[0]

    if stop_condition_pyramid_counter and stop_condition_tile_resolution:
        stop_condition_tile_resolution = False
        stop_condition_pyramid_counter = True
        
        if max_pyramid_depth is None:
            max_pyramid_depth = 0

    if stop_condition_pyramid_counter and not max_pyramid_depth:
        max_pyramid_depth = np.inf
        if pyramid_tiles_per_axis is not None:
            max_pyramid_depth = len(pyramid_tiles_per_axis)
        if pyramid_resolutions is not None:
            max_pyramid_depth = min(max_pyramid_depth, len(pyramid_resolutions))
        if max_pyramid_depth == np.inf:
            raise Exception('Passed arguments are incompatible.')
        if verbose:
            print(f'Maximum pyramid depth: {max_pyramid_depth}.')
    idx = 0
    
    if verbose:
        print('Stop conditions:')
        print(f'stop_condiction_tile_resolution: {stop_condition_tile_resolution}')
        print(f'stop_condition_pyramid_counter: {stop_condition_pyramid_counter}')
        print(f'Max pyramid depth: {max_pyramid_depth}')
    while True:
        if verbose:
            print(f'Next loop: {loop_counter}')
        
        if stop_condition_pyramid_counter and loop_counter > max_pyramid_depth:
            if verbose:
                print(f'Aborting loop because maximum pyramid depth ({max_pyramid_depth}) has been reached.')
            break

        ptnr_opts = deepcopy(nrpt_tile_options)
        
        if pyramid_resolutions is not None:
            if idx >= len(pyramid_resolutions):
                if verbose:
                    print(f'Maximum depth of pyramid resolution reached: {idx}. Breaking out of loop...')
                break
            res = pyramid_resolutions[idx]
            idx += 1
            if res == 0:
                if verbose:
                    print('Skipping this iteration, because pyramid resolution is 0.')
                loop_counter += 1
                continue
            ptnr_opts.nonrigid_registration_options.resolution = (res, res)
            
        # If loop_counter == 0 just do a normal nonrigid registration, since this will at least ensure
        # that both images have the same shape.
        
        n_tiles = int(2**loop_counter) if pyramid_tiles_per_axis is None else pyramid_tiles_per_axis[idx]
        if n_tiles == 0:
            if verbose:
                print(f'Skipping layer because number of tiles is 0.')
            loop_counter += 1
            continue
        if verbose:
            print(f'Number of tiles: {n_tiles}')
        tile_overlap_value = tile_overlap[idx] if isinstance(tile_overlap, list) else tile_overlap
        x_tile_props = get_tile_params(temp_image.shape[0], n_tiles = n_tiles, overlap = tile_overlap_value)
        y_tile_props = get_tile_params(temp_image.shape[1], n_tiles = n_tiles, overlap = tile_overlap_value)    
        
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
            
        ptnr_reg = register_tiles(moving_image_tiles, fixed_image_tiles, ptnr_opts, verbose)
        reg_results.append(ptnr_reg)
        
        if verbose:
            print('Tiles registered.')
        temp_image = transform_image(temp_image, ptnr_reg.registration.forward_transform, 'LINEAR')
        temp_images.append(temp_image)
        
        loop_counter += 1    
    final_reg_result = compose_registration_results(reg_results)
    if verbose:
        final_reg_result.registration.reg_params = {'pyramid_results': reg_results, 'temp_images': temp_images}
    return final_reg_result


def register(moving_img: numpy.ndarray,
             fixed_img: numpy.ndarray,
             moving_img_mask: numpy.ndarray | None = None,
             fixed_img_mask: numpy.ndarray | None = None,
             options: RegistrationOptions | None = None,
             verbose: bool = False) -> 'RegistrationResult':
    """Performs pairwise registration from moving_img to fixed_img. Optional tissue masks can be provided.
    Options are supplied via the options arguments.    

    Args:
        moving_img (numpy.ndarray): Source image.
        fixed_img (numpy.ndarray): Target image.
        moving_img_mask (Optional[numpy.ndarray], optional): Optional moving mask. Is otherwise derived automatically. Defaults to None.
        fixed_img_mask (Optional[numpy.ndarray], optional): Optional fixed mask. Is otherwise dervied automatically. Defaults to None.
        options (Optional[Options], optional): Can be supplied. Otherwise default arguments are used. Defaults to None.
        verbose (bool): Prints out more information. Defaults to False.

    Returns:
        RegistrationResult: Computed registration result.
    """
    # TODO: This function works, but should be cleaned up a bit more in the next update.
    if options is None:
        options = RegistrationOptions()
    reg_results = []
    do_tiling_reg = False
    if options.do_nonrigid_registration:
        if options.tiling_options.enable_tiling:
            # We set this to False now, since we are waiting to use tiling.
            options.do_nonrigid_registration = False
            do_tiling_reg = True
    if options.do_affine_registration or options.do_nonrigid_registration:
        reg_result = _perform_registration(moving_img,
                               fixed_img,
                               moving_img_mask,
                               fixed_img_mask,
                               options)
        reg_results.append(reg_result)
    # TODO: This should be used for nonrigid registration now.
    if do_tiling_reg:
        options.do_nonrigid_registration = True
        if verbose:
            print('Performing tiling registration.')
        if len(reg_results) > 0:
            warped_image = transform_image(moving_img, reg_results[0].registration.forward_transform)
        else:
            warped_image = moving_img.copy()
        if verbose:
            print('Doing simple tiling registration.')
        if options.tiling_options.tiling_mode == 'simple':
            tiling_reg_result = simple_tiling_registration(warped_image, 
                                                    fixed_img, 
                                                    options, 
                                                    verbose)            
        elif options.tiling_options.tiling_mode == 'pyramid':
            tiling_reg_result = pyramid_tiling_registration(warped_image, 
                                                    fixed_img, 
                                                    options, 
                                                    verbose)     
        else:
            raise Exception(f'Unkown tiling option: {options.tiling_options.tiling_mode}')
        reg_results.append(tiling_reg_result)
        reg_result = compose_registration_results(reg_results)
    else:
        reg_result = reg_results[0]
    # TODO: Clean this bit up. Its a bit unpredictable.
    if verbose:
        if reg_result.registration.reg_params is None:
            reg_params = {}
        else:
            reg_params = {'old_reg_params': reg_result.registration.reg_params}
        reg_params['aff_nr_reg_results'] = reg_results
        reg_result.registration.reg_params = reg_params
    return reg_result 

def _perform_registration(moving_img: numpy.ndarray,
              fixed_img: numpy.ndarray,
              moving_img_mask: numpy.ndarray | None = None,
              fixed_img_mask: numpy.ndarray | None = None,
              options: RegistrationOptions | None = None,                  
              **kwargs: dict) -> RegistrationResult:
    """Computes registration from moving image to fixed image. Performs preprocessing, 
    affine, deformable registration, and postprocessing.
    

    Args:
        moving_img (numpy.ndarray): Moving (or source) image.
        fixed_img (numpy.ndarray): Fixed (or target) image.
        moving_img_mask (Optional[numpy.ndarray], optional): Optional moving mask. Is otherwise dervied automatically. Defaults to None.
        fixed_img_mask (Optional[numpy.ndarray], optional): Optional fixed mask. Is otherwise dervied automatically. Defaults to None.
        options (Optional[RegistrationOptions], optional): Can be supplied. Otherwise default arguments are used. Defaults to None.

    Returns:
        RegistrationTransforms: 
    """
    if options is None:
        options = RegistrationOptions()

    # Setup tissue segmentation function.
    if options.disable_mask_generation or (moving_img_mask is not None and fixed_img_mask is not None):
        segmentation_function = None
    else:
        segmentation_function = load_segmentation_function(options.segmentation)
    
    # Setting up temporary directories.
    path_temp = options.temporary_directory
    clean_if_exists(path_temp)
    path_temp, _ = derive_subdir(path_temp)
    create_if_not_exists(path_temp)
    path_output = join(path_temp, 'output', 'registrations')
    path_output, subdir_num = derive_subdir(path_output)
    create_if_not_exists(path_output)
    
    # We use the paths dictionary to keep track of some paths.
    paths = {
        'path_temp': path_temp,
        'path_output': path_output,
    }
    
    path_metrics = os.path.join(path_output, 'metrics')
    path_metrics_small_resolution = os.path.join(path_metrics, 'small_resolution')
    path_metrics_full_resolution = os.path.join(path_metrics, 'full_resolution')

    create_if_not_exists(path_metrics_small_resolution)
    create_if_not_exists(path_metrics_full_resolution)

    paths['path_metrics'] = path_metrics
    paths['path_metrics_small_resolution'] = path_metrics_small_resolution
    paths['path_metrics_full_resolution'] = path_metrics_full_resolution
    paths['path_temp'] = path_temp    
    
    # Compute resampling factor for first resampling.
    max_size = max(moving_img.shape[0], moving_img.shape[1], fixed_img.shape[0], fixed_img.shape[1])            
    moving_resampling_factor, fixed_resampling_factor = derive_sampling_factors(
        options.pre_sampling_factor,
        max_size,
        options.pre_sampling_max_img_size
    )

    original_moving_image_size = moving_img.shape[:2]
    original_fixed_image_size = fixed_img.shape[:2]
    cmdln_returns = []

    # We collect some of the parameters used during registration for debugging. This should 
    # be moved to a logging file.
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

    # Convert to correct format, if necessary
    moving_img = correct_img_dtype(moving_img)
    fixed_img = correct_img_dtype(fixed_img)
    
    # First round of downsampling. This step is mostly used for speeding up the 
    # preprocessing. 
    moving_img = resample_image_sitk(moving_img, moving_resampling_factor)
    fixed_img = resample_image_sitk(fixed_img, fixed_resampling_factor)

    # Apply segmentation.
    if options.disable_mask_generation and moving_img_mask is None:
        moving_img_mask = np.ones(moving_img.shape[:2], dtype=np.uint8)
    elif moving_img_mask is None:
        moving_img_mask = segmentation_function(moving_img)
    else:
        moving_img_mask = resample_image_sitk(moving_img_mask, moving_resampling_factor)

    if options.disable_mask_generation and fixed_img_mask is None:            
        fixed_img_mask = np.ones(fixed_img.shape[:2], dtype=np.uint8)
    elif fixed_img_mask is None:
        fixed_img_mask = segmentation_function(fixed_img)
    else:
        fixed_img_mask = resample_image_sitk(fixed_img_mask, fixed_resampling_factor)

    moving_preprocessing_params['moving_img_mask'] = moving_img_mask
    fixed_preprocessing_params['fixed_img_mask'] = fixed_img_mask
    
    # Cropping and Padding of image and masks to make moving and fixed image/mask
    # have the same uniform shame.
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


    # Do affine registration.
    if options.do_affine_registration:
        moving_affine_path = join(path_temp, 'affine_moving_preprocessing')
        fixed_affine_path = join(path_temp, 'affine_fixed_preprocessing')
        moving_img_preprocessed = _preprocessing(moving_img,
                                                options.affine_registration_options.preprocessing_options,
                                                options.affine_registration_options.resolution,
                                                options.affine_registration_options.kernel_size,
                                                moving_affine_path,
                                                options.affine_registration_options.preprocessing_options.disable_denoising_moving)
        fixed_img_preprocessed = _preprocessing(fixed_img,
                                                options.affine_registration_options.preprocessing_options,
                                                options.affine_registration_options.resolution,
                                                options.affine_registration_options.kernel_size,
                                                fixed_affine_path,
                                                options.affine_registration_options.preprocessing_options.disable_denoising_fixed)
        height = fixed_img_preprocessed.height
        
        ia_init = ''
        if options.affine_registration_options.ia == 'ia-com-init' and fixed_img_mask is not None and moving_img_mask is not None:
            # Use Segmentation masks to compute center of mass initialization
            init_mat_path = os.path.join(path_temp, 'Affine_init.mat')
            init_mat = com_affine_matrix(moving_img, fixed_img_mask)
            write_mat_to_file(init_mat, init_mat_path)
            ia_init = ['-ia', f'{init_mat_path}']
            reg_params['com_x'] = init_mat[0, 2]
            reg_params['com_y'] = init_mat[1, 2]
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
        aff_ret = affine_registration(options.path_to_greedy,
                                        fixed_img_preprocessed.image_path,
                                        moving_img_preprocessed.image_path,
                                        path_small_affine,
                                        offset,
                                        ia_init,
                                        options.affine_registration_options,
                                        options.use_docker_container,
                                        path_temp
                                        )
        cmdln_returns.append(aff_ret)    
    else:
        path_small_affine = None  
    paths['path_small_affine'] = path_small_affine

    if options.do_nonrigid_registration:
        # Deformable registration is outsourced.
        # TODO: Clean this up. Either make the affine code like the deformable or the other way around.
        (def_reg_ret, 
            new_paths,
            moving_img_preprocessed,
            fixed_img_preprocessed) = _do_deformable_registration(moving_img,
                                        fixed_img,
                                        moving_img_mask,
                                        fixed_img_mask,
                                        options,
                                        paths,
                                        reg_params)
        cmdln_returns.append(def_reg_ret)
        for key in new_paths:
            paths[key] = new_paths[key]

    else:
        paths['path_small_warp'] = None
        paths['path_small_warp_inv'] = None      

    # Postprocessing. Takes care of rescaling and compositing transformations.
    registration_transform = _reg_postprocess(moving_img_preprocessed,
                            fixed_img_preprocessed,
                            moving_preprocessing_params,
                            fixed_preprocessing_params,
                            options,
                            moving_img,
                            reg_params,
                            paths)
    
    # An optional step for computing the reverse registration, i.e. from fixed image to moving.
    # Inverting the affine registration is not much of an issue, but the non-rigid registration
    # might need to be computed again. Typically not used. 
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
                fixed_img_preprocessed) = _do_deformable_registration(moving_img,
                                            fixed_img,
                                            moving_img_mask,
                                            fixed_img_mask,
                                            options,
                                            paths_rev,
                                            reg_params,
                                            compute_reverse_nonrigid_registration=True)
            cmdln_returns.append(def_reg_ret)      
            for key in new_paths:
                paths_rev[key] = new_paths[key]                  
        rev_registration_transform = _reg_postprocess(moving_img_preprocessed,
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
        __cleanup_temporary_directory(options.temporary_directory)  
    registration_result = RegistrationResult(registration_transform, rev_registration_transform)      
    return registration_result


def _do_deformable_registration(moving_img: numpy.ndarray,
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
    # If we use a different resolution during nonrigid registration,
    # we rescale images and transformation here.
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
    
    # Diffeomorphic registration.
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


    moving_img_preprocessed = _preprocessing(moving_img,
                                            options.nonrigid_registration_options.preprocessing_options,
                                            options.nonrigid_registration_options.resolution,
                                            options.nonrigid_registration_options.kernel_size,
                                            moving_nr_path,
                                            options.nonrigid_registration_options.preprocessing_options.disable_denoising_moving)
    fixed_img_preprocessed = _preprocessing(fixed_img,
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

    deformable_reg_ret = deformable_registration(options.path_to_greedy,
                                                    fixed_img_preprocessed.image_path,
                                                    moving_img_preprocessed.image_path,
                                                    options.nonrigid_registration_options,
                                                    output_warp=path_small_warp,
                                                    output_inv_warp=path_small_warp_inv,
                                                    affine_pre_transform=nonrigid_affine_trans_path,
                                                    ia=ia_init,
                                                    use_docker_container=options.use_docker_container,
                                                    temp_directory=path_temp)
    return deformable_reg_ret, new_paths, moving_img_preprocessed, fixed_img_preprocessed
        

def _reg_postprocess(moving_img_preprocessed: PreprocessedData,
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
    # TODO: This needs some changing. There should be an option not to composite affine and nonrigid registrations, so that 
    # affine keeps being unbounded. Some of the options could also be removed. No point in having the affine registration be 
    # bounded anyway. 
    if options.do_nonrigid_registration:
        path_metrics_small_resolution = paths['path_metrics_small_resolution']
        path_metrics_full_resolution = paths['path_metrics_full_resolution']
        path_small_warp = paths['path_small_warp']
        path_small_warp_inv = paths['path_small_warp_inv']
        no_2_orig_resample = (options.nonrigid_registration_options.resolution[0] / moving_img.shape[0]) * 100
        no_2_orig_factor = 100 / no_2_orig_resample            
        if options.do_affine_registration and not options.affine_registration_options.keep_affine_transform_unbounded:
            # TODO: This option can be removed as there is no point in removing the unbounded properties of the affine
            # transformation here.
            path_small_composite_warp = os.path.join(path_metrics_small_resolution, 'small_composite_warp.nii.gz')
            nonrigid_affine_trans_path = paths['nonrigid_affine_trans_path']
            composite_warps(
                options.path_to_greedy,
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
                options.path_to_greedy,
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
        cmdl_log=None,
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
    registration_result = RegistrationTransforms(forward_transform=fixed_transform, backward_transform=moving_transform, reg_params=reg_result)
    # Return this!
    return registration_result


def _reg_aff_grp_wrapper(fixed_tuple: tuple[numpy.ndarray, numpy.ndarray | None], 
                        moving_tuple: tuple[numpy.ndarray, numpy.ndarray | None], 
                        options: RegistrationOptions, 
                        idx: int) -> tuple[RegistrationTransforms, RegistrationTransforms, int]:
    """Wrapper function for setting up arguments so that we can use multiprocessing.

    Args:
        fixed_tuple (tuple[numpy.ndarray, numpy.ndarray | None]): Tuple of fixed image and fixed mask.
        moving_tuple (tuple[numpy.ndarray, numpy.ndarray | None]): Tuple of moving image and moving mask.
        options (RegistrationOptions): RegistrationOptions.
        idx (int): Idx. Used for ensuring that we have the right registration order after multiprocessing finished.

    Returns:
        tuple[RegistrationTransforms, RegistrationTransforms, int]: Registration, reverse registration and integer indicating order.
    """
    fixed_image, fixed_mask = fixed_tuple
    moving_image, moving_mask = moving_tuple
    reg_result = _perform_registration(
        moving_img=moving_image,
        fixed_img=fixed_image,
        moving_img_mask=moving_mask,
        fixed_img_mask=fixed_mask,
        options=options        
    )
    return reg_result.registration, reg_result.reverse_registration, idx


def _reg_nr_grp_wrapper(fixed_tuple: tuple[numpy.ndarray, numpy.ndarray | None], 
                       moving_tuple: tuple[numpy.ndarray, numpy.ndarray | None], 
                       options: RegistrationOptions, 
                       composited_fixed_transform: sitk.Transform, 
                       idx: int) -> tuple[RegistrationTransforms, RegistrationTransforms, int]:
    """Wrapper function for executing nonrigid registration in groupwise registration modus
    with multiprocessing.

    Args:
        fixed_tuple (tuple[numpy.ndarray, numpy.ndarray  |  None]): Fixed image data.
        moving_tuple (tuple[numpy.ndarray, numpy.ndarray  |  None]): Moving image data.
        options (RegistrationOptions): Options.
        composited_fixed_transform (sitk.Transform): Prealignment transform. Results of affine registration.
        idx (int): Integer indicating the order during multiprocessing.

    Returns:
        tuple[RegistrationTransforms, RegistrationTransforms, int]: _description_
    """
    fixed_image, fixed_mask = fixed_tuple
    moving_image, moving_mask = moving_tuple
    warped_image = transform_image(moving_image, composited_fixed_transform, 'LINEAR')
    if moving_mask is not None:
        warped_mask = transform_image(moving_mask, composited_fixed_transform, 'NN')
    else:
        warped_mask = None
    # print(f'Temporary directory in wrapper function: {options.temporary_directory}')
    nonrigid_reg_result = _perform_registration(moving_img=warped_image, 
                                    fixed_img=fixed_image, 
                                    moving_img_mask=warped_mask, 
                                    fixed_img_mask=fixed_mask, 
                                    options=options)
    deformable_warped_image = transform_image(warped_image, nonrigid_reg_result.registration.forward_transform, 'LINEAR')
    return nonrigid_reg_result.registration, nonrigid_reg_result.reverse_registration, deformable_warped_image, idx


# TODO: Add option for skipping affine registration.
def groupwise_registration(image_mask_list: list[tuple[numpy.ndarray, numpy.ndarray | None]],
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
        image_mask_list (List[Tuple[numpy.ndarray, Optional[numpy.ndarray]]]): List of images. Last image is fixed image. 
            Every other image is a moving image. For each image, an optional mask can be supplied.
        options (Optional[RegistrationOptions], optional): Registration options. At this moment, the affine registration is
        always executed. `options.do_affine_registration` is ignored, but the nonrigid registration can be disabled.
        Defaults to None.

    Returns:
        Tuple[GroupwiseRegResult, List[numpy.ndarray]]: GroupwiseRegResult contains all computed transformations. List of images are either affine or nonrigid warped images.
    """
    if options is None:
        options = RegistrationOptions()
    segmentation_function = load_segmentation_function(options.segmentation)
    # Stage 1: Prepare masks if necessary.
    new_image_mask_list = []
    for tpl in image_mask_list:
        if isinstance(tpl, tuple):
            if len(tpl) == 2:
                img, mask = tpl
            else:
                img = tpl[0]
                mask = None
        else:
            img = tpl
            mask = None
        if mask is None and not options.disable_mask_generation:
            mask = segmentation_function(img)
        new_image_mask_list.append((img, mask))
    image_mask_list = new_image_mask_list
    do_nr_registration = options.do_nonrigid_registration
    # Stage1: Affine register along the the sequence.
    moving_image, moving_mask = image_mask_list[0]
    # We set this to false for the affine part and set it back to the original value afterwards.
    options.do_nonrigid_registration = False
    options.do_affine_registration = True
    affine_transform_lists = []    
    reverse_affine_transform_list = []
    if options.grp_n_proc is None:
        for fixed_tuple in image_mask_list[1:]:
            fixed_image, fixed_mask = fixed_tuple
            reg_result = _perform_registration(moving_image, fixed_image, moving_mask, fixed_mask, options)
            affine_transform_lists.append(reg_result.registration)
            reverse_affine_transform_list.append(reg_result.reverse_registration)
            moving_image = fixed_image
            moving_mask = fixed_mask
    else:
        pargs = []
        for idx in range(len(image_mask_list) - 1):
            imm = image_mask_list[idx]
            imf = image_mask_list[idx + 1]
            options_ = deepcopy(options)
            options_.temporary_directory = join(options_.temporary_directory, f'gfh_grp_{idx}')
            pargs.append((imf, imm, options_, idx))
        pool = multiprocess.Pool(options.grp_n_proc)
        ret = pool.starmap(_reg_aff_grp_wrapper, pargs)
        ret.sort(key=lambda x: x[2])
        for ret_ in ret:
            affine_transform_lists.append(ret_[0])
            reverse_affine_transform_list.append(ret_[1])
    
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
    fixed_image, fixed_mask = fixed_tuple
    if options.grp_n_proc is None:
        for idx, moving_tuple in enumerate(image_mask_list[:-1]):
            moving_image, moving_mask = moving_tuple
            composited_fixed_transform = compose_transforms([x.forward_transform for x in affine_transform_lists][idx:])
            warped_image = transform_image(moving_image, composited_fixed_transform, 'LINEAR')
            warped_mask = transform_image(moving_mask, composited_fixed_transform, 'NN')
            nonrigid_reg_result = register(warped_image, fixed_image, warped_mask, fixed_mask, options=options)
            deformable_warped_image = transform_image(warped_image, nonrigid_reg_result.registration.forward_transform, 'LINEAR')
            nonrigid_warped_images.append(deformable_warped_image)
            nonrigid_transformations.append(nonrigid_reg_result.registration)
            reverse_nonrigid_transformations.append(nonrigid_reg_result.reverse_registration)
    else:
        nr_pairs = []
        for idx, moving_tuple in enumerate(image_mask_list[:-1]):
            moving_image, moving_mask = moving_tuple
            composited_fixed_transform = compose_transforms([x.forward_transform for x in affine_transform_lists][idx:])
            options_ = deepcopy(options)
            tmp_dir = f'grp_nr_{idx}'            
            options_.temporary_directory = join(options.temporary_directory, tmp_dir)
            nr_pairs.append(((fixed_image, fixed_mask), (moving_image, moving_mask), options_, composited_fixed_transform, idx))
        pool = multiprocess.Pool(options.grp_n_proc)
        ret = pool.starmap(_reg_nr_grp_wrapper, nr_pairs)
        ret.sort(key=lambda x: x[-1])
        for ret_ in ret:
            nonrigid_transformations.append(ret_[0])
            reverse_nonrigid_transformations.append(ret_[1])        
            nonrigid_warped_images.append(ret_[2])
            
    reverse_nonrigid_transformations = reverse_nonrigid_transformations[::-1]
    groupwise_registration_results = GroupwiseRegResult(affine_transform_lists, 
                                                        reverse_affine_transform_list,
                                                        nonrigid_transformations,
                                                        reverse_nonrigid_transformations)
    return groupwise_registration_results, nonrigid_warped_images


def transform_image(image: numpy.ndarray,
                    transform: GFHTransform | RegistrationTransforms | RegistrationResult,
                    interpolation_mode: str | int = 'LINEAR') -> numpy.ndarray:
    """Transforms image data from moving to fixed image space using computed transformation. Use forward_transform attribute.

    Args:
        image (numpy.ndarray): 
        transform (GFHTransform): 
        interpolation_mode (str, optional): Defaults to 'LINEAR'.

    Returns:
        numpy.ndarray: 
    """
    if isinstance(transform, RegistrationResult):
        transform = transform.registration
    if isinstance(transform, RegistrationTransforms):
        transform = transform.forward_transform    
    return _transform_image(image,
                            transform.transform,
                            transform.size,
                            interpolation_mode)


def _transform_image(image: numpy.ndarray, 
                     transform: SimpleITK.SimpleITK.Transform,
                     size: tuple[int,int],
                     interpolation_mode: str | int = 'LINEAR') -> numpy.ndarray:
    """Transforms image from moving to fixed image space.

    Args:
        image (numpy.ndarray): 
        transform (SimpleITK.SimpleITK.Transform): 
        size (Tuple[int,int]): Fixed image space resolution.
        interpolation_mode (str, optional): 'LINEAR' or 'NN'. Defaults to 'LINEAR'.

    Returns:
        numpy.ndarray: _description_
    """
    if isinstance(interpolation_mode, str):
        if interpolation_mode == 'LINEAR':
            interpolation_mode = sitk.sitkLinear
        else:
            interpolation_mode = sitk.sitkNearestNeighbor
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


def transform_pointset(pointset: numpy.ndarray,
                       transform: GFHTransform | RegistrationTransforms | RegistrationResult | SimpleITK.Transform) -> numpy.ndarray:
    """Transforms pointset from moving to fixed image space. Use backward_transform attribute.

    Args:
        pointset (numpy.ndarray): 
        transform (GFHTransform): 

    Returns:
        numpy.ndarray:
    """
    if isinstance(transform, RegistrationResult):
        transform = transform.registration
    if isinstance(transform, RegistrationTransforms):
        transform = transform.forward_transform
    if isinstance(transform, GFHTransform):
        transform = transform.transform
    return _transform_pointset(
        pointset,
        transform
    )


def _transform_pointset(pointset: numpy.ndarray,
                        transform: SimpleITK.Transform) -> numpy.ndarray:
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
def transform_geojson(geojson_data: geojson.GeoJSON,
                      transform: RegistrationResult | RegistrationTransforms | GFHTransform | SimpleITK.Transform,
                      **kwards) -> list[geojson.GeoJSON] | geojson.GeoJSON:
    """Applies transformation to geojson data. Can be a feature collection ot a list of features.

    Args:
        geojson_data (geojson.GeoJSON): 
        transformation (SimpleITK.SimpleITK.Image): 

    Returns:
        list[geojson.GeoJSON] | geojson.GeoJSON: 
    """
    if isinstance(transform, RegistrationResult):
        transform = transform.registration
    if isinstance(transform, RegistrationTransforms):
        transform = transform.forward_transform
    if isinstance(transform, GFHTransform):
        transform = transform.transform    
    if not isinstance(geojson_data, list):
        geometries = geojson_data['features']
    else:
        geometries = geojson_data
    warped_geometries = []
    for _, geometry in enumerate(geometries):
        warped_geometry = geojson.utils.map_tuples(lambda coords: __warp_geojson_coord_tuple(coords, transform), geometry)
        warped_geometries.append(warped_geometry)
    if not isinstance(geojson_data, list):
        geojson_data['features'] = warped_geometries
        return geojson_data
    else:
        return warped_geometries


def __warp_geojson_coord_tuple(coord: tuple[float, float], transform: SimpleITK.Transform) -> tuple[float, float]:
    """Transforms coordinates from geojson data from moving to fixed image space.

    Args:
        coord (Tuple[float, float]): 
        transform (SimpleITK.SimpleITK.Transform): 

    Returns:
        Tuple[float, float]: 
    """
    ps = np.array([[coord[0], coord[1]]]).astype(float)
    warped_ps = transform_pointset(ps, transform)
    return (warped_ps[0, 0], warped_ps[0, 1])


def __cleanup_temporary_directory(directory: str) -> None:
    """Removes the temporary directory.

    Args:
        directory (str):
    """
    shutil.rmtree(directory)


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
        
    use_docker_container: bool
        If GreedyFHist is called from outside a docker container, but greedy is inside 
        a docker container, set this option to True and set the image name with greedy
        as `path_to_greedy`.

    segmentation_function: Optional[Callable]
        Segmentation function for foreground segmentation.

    """

    name: str = 'GreedyFHist'
    path_to_greedy: str = 'greedy'
    use_docker_container: bool = False
    segmentation: SegmentationOptions | Callable[[numpy.ndarray], numpy.ndarray] | str | None = None

    def __post_init__(self):
        if isinstance(self.segmentation, SegmentationOptions) or isinstance(self.segmentation, str):
            self.segmentation = load_segmentation_function(self.segmentation)
        if self.segmentation is None:
            self.segmentation = load_segmentation_function('yolo-seg')
            
    def __update_options(self, 
                         options: RegistrationOptions,
                         skip_segmentation_assignment: bool = False):
        if self.path_to_greedy is not None:
            options.path_to_greedy = self.path_to_greedy
        if self.segmentation is not None and not skip_segmentation_assignment:
            options.segmentation = self.segmentation
        options.use_docker_container = self.use_docker_container
        return options        

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
        if options is None:
            options = RegistrationOptions()
        options = self.__update_options(options)
        return register_from_filepaths(
            moving_img_path=moving_img_path,
            fixed_img_path=fixed_img_path,
            target_img_path=target_img_path,
            moving_img_mask_path=moving_img_mask_path,
            fixed_img_mask_path=fixed_img_mask_path,
            options=options,
            transform_path=transform_path
            )

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
        if options is None:
            options = RegistrationOptions()
        options = self.__update_options(options)        
        return groupwise_registration_from_filepaths(
            image_mask_filepaths=image_mask_filepaths,
            target_directory=target_directory,
            options=options
        )

    def tiling_registration(self,
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
        registration_options = self.__update_options(registration_options, skip_segmentation_assignment=True)
        return simple_tiling_registration(
            moving_image=moving_image,
            fixed_image=fixed_image,
            registration_options=registration_options,
            verbose=verbose
        )

    def pyramid_tiling_registration(self,
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
        registration_options = self.__update_options(registration_options, skip_segmentation_assignment=True)
        return pyramid_tiling_registration(
            moving_image=moving_image,
            fixed_image=fixed_image,
            registration_options=registration_options,
            verbose=verbose
        )

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
            RegistrationResult: Computed registration result.
        """
        # TODO: This function works, but should be cleaned up a bit more in the next update.
        options = self.__update_options(options)
        return register(
            moving_img=moving_img,
            fixed_img=fixed_img,
            moving_img_mask=moving_img_mask,
            fixed_img_mask=fixed_img_mask,
            options=options,
            verbose=verbose
        )

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
            image_mask_list (List[Tuple[numpy.ndarray, Optional[numpy.ndarray]]]): 
                List of input images. The last image is denoted as the fixed image. Every other image is treated
                as a moving image. For each image, an optionsl mask can be supplied.
                
            options (RegistrationOptions, optional) = None:
                Registration options. If None, uses default options. Defaults to None.
                
        Returns:
            Tuple[GroupwiseRegResult, List[numpy.ndarray]]: GroupwiseRegResult contains all computed transformations. 
                List of images are either affine or nonrigid warped images.
        """
        if options is None:
            options = RegistrationOptions()
        options = self.__update_options(options)
        return groupwise_registration(
            image_mask_list=image_mask_list,
            options=options
        )

    def transform_image(self,
                        image: numpy.ndarray,
                        transform: GFHTransform | RegistrationTransforms | RegistrationResult,
                        interpolation_mode: str | int = 'LINEAR') -> numpy.ndarray:
        """Transforms image data from moving to fixed image space using computed transformation. Use forward_transform attribute.

        Args:
            image (numpy.ndarray): 
            transform (GFHTransform | RegistrationTransforms | RegistrationResult): Transform to use. 
                By default tries to look for the forward transformation. 
            interpolation_mode (str, optional): Defaults to 'LINEAR'.

        Returns:
            numpy.ndarray: 
        """
        return transform_image(
            image=image,
            transform=transform,
            interpolation_mode=interpolation_mode
        )

    def transform_pointset(self,
                           pointset: numpy.ndarray,
                           transform: RegistrationResult | RegistrationTransforms | GFHTransform | SimpleITK.Transform) -> numpy.ndarray:
        """Transforms pointset from moving to fixed image space. Use backward_transform attribute.

        Args:
            pointset (numpy.ndarray): 
            transform (GFHTransform): 

        Returns:
            numpy.ndarray:
        """
        return transform_pointset(
            pointset=pointset,
            transform=transform
        )

    # TODO: Fix types for geojson.
    def transform_geojson(self,
                          geojson_data: geojson.GeoJSON,
                          transformation: RegistrationResult | RegistrationTransforms | GFHTransform | SimpleITK.Transform,
                          **kwards) -> list[geojson.GeoJSON] | geojson.GeoJSON:
        """Applies transformation to geojson data. Can be a feature collection ot a list of features.

        Args:
            geojson_data (geojson.GeoJSON): 
            transformation (RegistrationResult | RegistrationTransforms | GFHTransform | SimpleITK.Transform): 

        Returns:
            list[geojson.GeoJSON] | geojson.GeoJSON: 
        """
        return transform_geojson(geojson_data=geojson_data,
                                 transformation=transformation)