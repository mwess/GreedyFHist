"""
Utils for transformation of image and warp functions.
"""
import os

from typing import Optional, Tuple, Union

import cv2
import numpy
import numpy as np
import scipy.ndimage as nd
import SimpleITK
import SimpleITK as sitk
from skimage.transform import resize
from skimage.filters import gaussian
from skimage.color import rgb2gray

from greedyfhist.utils.utils import call_command
from greedyfhist.custom_types import padding_type, image_shape

def com_affine_matrix(fixed: numpy.array, moving: numpy.array) -> numpy.array:
    """
    Compute the difference in center of mass between fixed and moving image masks.
    """
    mat = numpy.eye(3)
    fixed_com = nd.center_of_mass(fixed)
    moving_com = nd.center_of_mass(moving)
    mat[0, 2] = fixed_com[0] - moving_com[0]
    mat[1, 2] = fixed_com[1] - moving_com[1]
    return mat

# Rescale functions
def rescale_affine(small_affine_path: str, big_affine_path: str, factor: float) -> None:
    with open(small_affine_path) as f:
        my_var = list(map(float, f.read().split()))
    # Modify translation vector
    new_val_1 = my_var[2] * factor
    new_val_2 = my_var[5] * factor
    out_str = f'{my_var[0]} {my_var[1]} {new_val_1}\n{my_var[3]} {my_var[4]} {new_val_2}\n{my_var[6]} {my_var[7]} {my_var[8]}'
    with open(big_affine_path, 'w') as f:
        f.write(out_str)


def rescale_warp_test(path_to_c2d: str, 
                      small_warp_path: str, 
                      big_warp_path: str, 
                      small_resolution: image_shape, 
                      original_resolution_padded: image_shape,
                      original_resolution: image_shape, 
                      factor: float, 
                      temp_path: Optional[str] =None, 
                      cleanup: bool =True):
    # create a blank image with 1's as background.

    WIDTH_small, HEIGHT_small = small_resolution
    WIDTH_original, HEIGHT_original = original_resolution
    WIDTH_original_padded, HEIGHT_original_padded = original_resolution_padded
    # Make sure that this only affects the last part of the file
    warp_without_nii_path = small_warp_path.replace('.nii.gz', '')
    if temp_path is not None:
        base_name = os.path.basename(warp_without_nii_path)
        warp_without_nii_path = os.path.join(temp_path, base_name)

    cmdln_returns = []

    # create a blank image with 1's as background.
    PATH_mask_no_pad = warp_without_nii_path + '_mask_no_pad.nii.gz'
    cmd = f'{path_to_c2d} -background 1 -create {WIDTH_small}x{HEIGHT_small} 1x1mm -orient LP -o {PATH_mask_no_pad}'
    ret = call_command(cmd)
    cmdln_returns.append(ret)

    # Pad created image to "padding size" and add 0 as padding.
    PATH_mask_padded = warp_without_nii_path + '_mask_padded.nii.gz'
    cmd = f'{path_to_c2d} {PATH_mask_no_pad} -pad-to {WIDTH_original_padded}x{HEIGHT_original_padded} 0 -o {PATH_mask_padded}'
    ret = call_command(cmd)
    cmdln_returns.append(ret)

    # Set origin of image to 0x0
    cmd = f'{path_to_c2d} {PATH_mask_padded} -origin 0x0mm -o {PATH_mask_padded}'
    ret = call_command(cmd)
    cmdln_returns.append(ret)

    # Set origin of warp file to 0x0
    cmd = f'{path_to_c2d} -mcs {small_warp_path} -origin 0x0mm -omc {small_warp_path}'
    ret = call_command(cmd)
    cmdln_returns.append(ret)

    # Add mask to image
    PATH_small_warp_no_pad = warp_without_nii_path + '_small_warp_no_pad.nii.gz'
    cmd = f'{path_to_c2d} -mcs {PATH_mask_padded} -popas mask -mcs {small_warp_path} -foreach -push mask -add -endfor -omc {PATH_small_warp_no_pad}'
    ret = call_command(cmd)
    cmdln_returns.append(ret)

    # Multiply mask with small warp. (Treat warp like an image and cut off from the edges of the warp)
    PATH_small_warp_no_pad = warp_without_nii_path + '_small_warp_no_pad.nii.gz'
    cmd = f'{path_to_c2d} -mcs {PATH_mask_padded} -popas mask -mcs {PATH_small_warp_no_pad} -foreach -push mask -times -endfor -omc {PATH_small_warp_no_pad}'
    ret = call_command(cmd)
    cmdln_returns.append(ret)

    # Remove padded area
    PATH_small_warp_no_pad_trim = warp_without_nii_path + '_small_warp_no_pad_trim.nii.gz'
    cmd = f'{path_to_c2d} -mcs {PATH_small_warp_no_pad} -foreach -trim 0vox -endfor -omc {PATH_small_warp_no_pad_trim}'
    ret = call_command(cmd)
    cmdln_returns.append(ret)

    # Subtract mask from trimmed image
    PATH_small_warp_no_pad = warp_without_nii_path + '_small_warp_no_pad.nii.gz'
    cmd = f'{path_to_c2d} -mcs {PATH_mask_no_pad} -popas mask -mcs {PATH_small_warp_no_pad_trim} -foreach -push mask -scale -1 -add -endfor -omc {PATH_small_warp_no_pad_trim}'
    ret = call_command(cmd)
    cmdln_returns.append(ret)

    # Resample warp to original dimensions
    # PATH_big_warp = os.path.join(PATH_metrics_full_resolution, 'big_warp.nii.gz')
    cmd = f'{path_to_c2d} -mcs {PATH_small_warp_no_pad_trim} -foreach -resample {WIDTH_original}x{HEIGHT_original} -scale {factor} -spacing 1x1mm -origin 0x0mm -endfor -omc {big_warp_path}'
    ret = call_command(cmd)
    cmdln_returns.append(ret)

    return cmdln_returns

def rescale_warp(small_warp_path: str,
                 big_warp_path: str,
                 small_resolution: image_shape,
                 original_resolution: image_shape,
                 factor: float):
    warp = sitk.GetArrayFromImage(sitk.ReadImage(small_warp_path))
    mask = np.ones((small_resolution[0], small_resolution[1], 2), dtype=np.uint8)
    padding = (warp.shape[0] - small_resolution[0])//2
    mask = np.pad(mask, ((padding, padding), (padding, padding), (0, 0)))
    warp_no_pad = (warp + mask)[padding:-padding, padding:-padding]
    big_warp = resize(warp_no_pad, (original_resolution[0], original_resolution[1])) * factor
    big_warp_sitk = sitk.GetImageFromArray(big_warp, isVector=True)
    sitk.WriteImage(big_warp_sitk, big_warp_path)

def apply_mask(image: numpy.array, mask: numpy.array) -> numpy.array:
    return image * np.expand_dims(mask, -1).astype(np.uint8)


def pad_pointcloud(pc, padding: padding_type):
    left, _, top, _ = padding
    pc.x = pc.x + left
    pc.y = pc.y + top
    return pc


def rescale_image(image: numpy.array, resolution: image_shape) -> numpy.array:
    return cv2.resize(image, (resolution[1], resolution[0]))


def get_symmetric_padding(img1: numpy.array, img2: numpy.array) -> Tuple[padding_type, padding_type]:
    max_size = max(img1.shape[0], img1.shape[1], img2.shape[0], img2.shape[1])
    # print(max_size)
    padding_img1 = get_padding_params(img1, max_size)
    padding_img2 = get_padding_params(img2, max_size)
    return padding_img1, padding_img2


def get_padding_params(img: numpy.array, shape: image_shape):
    pad_x = shape - img.shape[0]
    pad_x_l = pad_x // 2
    pad_x_u = pad_x // 2
    if pad_x % 2 != 0:
        pad_x_u += 1
    pad_y = shape - img.shape[1]
    pad_y_l = pad_y // 2
    pad_y_u = pad_y // 2
    if pad_y % 2 != 0:
        pad_y_u += 1
    return pad_y_l, pad_y_u, pad_x_l, pad_x_u


def denoise_image(image: numpy.array, resolution: int =512, sp: int =20, sr: int =20, maxLevel: int =2) -> numpy.array:
    shape = image.shape
    img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    img_hsv = cv2.resize(img_hsv, (resolution, resolution))
    img_denoised = cv2.pyrMeanShiftFiltering(img_hsv, sp, sr, maxLevel)
    img_denoised = cv2.cvtColor(img_denoised, cv2.COLOR_HSV2RGB)
    img_denoised = cv2.resize(img_denoised, (shape[1], shape[0]))
    return img_denoised


def resample_image_with_gaussian(image: numpy.array, resolution: image_shape, sigma: float):
    image = gaussian(image, sigma, channel_axis=-1)
    image = resize(image, resolution)
    image = rgb2gray(image) * 255
    image = image.astype(np.float32)
    return image


def resample_image(path_to_c2d: str, image_path: str, out: str, resample_factor: float, smoothing: float =None):
    resampling_cmd = f''
    if smoothing is not None:
        resampling_cmd = resampling_cmd + f'-smooth-fast {smoothing}x{smoothing}vox '
    resampling_cmd = resampling_cmd + f'-resample {resample_factor}% -spacing 1x1mm -orient LP -origin 0x0mm -o'
    cmd = f'{path_to_c2d} {image_path} {resampling_cmd} {out}'
    ret = call_command(cmd)
    return ret


def pad_image(image: numpy.array, padding: padding_type, constant_values: float =0):
    dims = len(image.shape)
    if dims == 2:
        padded_image = np.pad(image, ((padding, padding), (padding, padding)), constant_values=constant_values)
    elif dims == 3:
        padded_image = np.pad(image, ((padding, padding), (padding, padding), (0,0)), constant_values=constant_values)
    else:
        pass # Throw error instead
    return padded_image


def get_symmetric_padding(img1: numpy.array, img2: numpy.array) -> Tuple[padding_type, padding_type]:
    max_size = max(img1.shape[0], img1.shape[1], img2.shape[0], img2.shape[1])
    padding_img1 = get_padding_params(img1, max_size)
    padding_img2 = get_padding_params(img2, max_size)
    return padding_img1, padding_img2


def get_padding_params(img: numpy.array, shape: int) -> padding_type:
    pad_x = shape - img.shape[0]
    pad_x_l = pad_x // 2
    pad_x_u = pad_x // 2
    if pad_x % 2 != 0:
        pad_x_u += 1
    pad_y = shape - img.shape[1]
    pad_y_l = pad_y // 2
    pad_y_u = pad_y // 2
    if pad_y % 2 != 0:
        pad_y_u += 1
    return pad_y_l, pad_y_u, pad_x_l, pad_x_u


def pad_asym(image: numpy.array, padding: padding_type, constant_values: int = 0) -> numpy.array:
        left, right, top, bottom = padding
        if len(image.shape) == 2:
            image = np.pad(image, ((top, bottom), (left, right)), constant_values=constant_values)
        else:
            # Assume 3 dimensions
            image = np.pad(image, ((top, bottom), (left, right), (0, 0)), constant_values=constant_values)
        return image


def remove_padding(image: numpy.array, padding: padding_type) -> numpy.array:
    left, right, top, bottom = padding
    bottom_idx = -bottom if bottom != 0 else image.shape[0]
    right_idx = -right if right != 0 else image.shape[1]
    return image[top:bottom_idx, left:right_idx]
    

def empty_image(resolution: padding_type, dtype=type) -> SimpleITK.SimpleITK.Image:
    img = np.zeros(resolution, dtype=dtype)
    sitk_img = sitk.GetImageFromArray(img)
    direction = tuple(map(lambda x: x*-1, sitk_img.GetDirection()))
    sitk_img.SetDirection(direction)
    # Do I need to set origin
    return sitk_img

def build_empty_ref_image(sitk_image: SimpleITK.SimpleITK.Image) -> SimpleITK.SimpleITK.Image:
    shape = (sitk_image.GetWidth(), sitk_image.GetHeight())
    origin = sitk_image.GetOrigin()
    spacing = sitk_image.GetSpacing()
    direction = sitk_image.GetDirection()
    empty_img = sitk.Image(shape, sitk_image.GetPixelIDValue())
    empty_img.SetOrigin(origin)
    empty_img.SetSpacing(spacing)
    empty_img.SetDirection(direction)
    return empty_img

def write_empty_ref_image_to_file(image: SimpleITK.SimpleITK.Image, path: str) -> None:
    empty_image = build_empty_ref_image(image)
    sitk.WriteImage(empty_image, path)

def cropping(mask: numpy.array) -> Tuple[numpy.array, padding_type]:
    p = np.argwhere(mask == 1)
    min_x = int(np.min(p[:,0]))
    max_x = int(np.max(p[:,0]))
    min_y = int(np.min(p[:,1]))
    max_y = int(np.max(p[:,1]))
    cropped_mask = mask[min_x:max_x, min_y:max_y]
    return cropped_mask, (min_x, max_x, min_y, max_y)    

def add_cropped_region(img: numpy.array, original_shape: image_shape, cropping: padding_type) -> numpy.array:
    rh_x = original_shape[0]-cropping[1]
    rh_y = original_shape[1]-cropping[3]
    if len(img.shape) == 2:
        img_uncropped = np.pad(img, ((cropping[0], rh_x), (cropping[2], rh_y)))
    else:
        img_uncropped = np.pad(img, ((cropping[0], rh_x), (cropping[2], rh_y), (0,0)))
    return img_uncropped

def resample_by_factor(img: numpy.array, factor: float) -> numpy.array:
    img2 = cv2.resize(img, (int(img.shape[1]//factor), int(img.shape[0]//factor)))
    return img2

# def resize_image(img: numpy.array, shape: image_shape, interpolation: str ='NN') -> numpy.array:
#     if interpolation == 'NN':
#         interpolation_mode = cv2.INTER_NEAREST
#     else:
#         interpolation_mode = cv2.INTER_LINEAR
#     return cv2.resize(img, (shape[1], shape[0]), interpolation_mode)

def resize_image(img: numpy.array, shape: image_shape, interpolation: str ='NN') -> numpy.array:
    if interpolation == 'NN':
        # interpolation_mode = cv2.INTER_NEAREST
        interpolation_mode = 0
    else:
        # interpolation_mode = cv2.INTER_LINEAR
        interpolation_mode = 1
    return resize(img, (shape[0], shape[1]), order=interpolation_mode)