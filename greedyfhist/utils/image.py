"""
Utils for transformation of image and warp functions.
"""
from typing import Any, Optional, Tuple

import cv2
import numpy
import numpy as np
import scipy.ndimage as nd
import SimpleITK
import SimpleITK as sitk
from skimage.transform import resize
from skimage.filters import gaussian
from skimage.color import rgb2gray
import tifffile

from greedyfhist.utils.utils import call_command
from greedyfhist.custom_types import padding_type, image_shape


def com_affine_matrix(fixed: numpy.array, moving: numpy.array) -> numpy.array:
    """
    Compute the difference in center of mass between fixed and moving image masks.

    Returns: Affine matrix
    """
    mat = numpy.eye(3)
    fixed_com = nd.center_of_mass(fixed)
    moving_com = nd.center_of_mass(moving)
    mat[0, 2] = fixed_com[0] - moving_com[0]
    mat[1, 2] = fixed_com[1] - moving_com[1]
    return mat


def get_com_offset(mat: numpy.array) -> float:
    """
    Computes the translation offset of the given matrix.
    
    Args:
        mat (numpy.array):
            transform matrix
            
    Returns:
        translation offset
    """
    translation = mat[:2, 2]
    return np.sqrt(np.square(translation[0]) + np.square(translation[1]))


def read_affine_transform(small_affine_path: str) -> SimpleITK.SimpleITK.Transform:
    with open(small_affine_path) as f:
        my_var = list(map(float, f.read().split()))
    # Modify translation vector

    affine_transform = sitk.AffineTransform(2)
    affine_transform.SetTranslation((my_var[2], my_var[5]))
    affine_transform.SetMatrix((my_var[0], my_var[1], my_var[3], my_var[4]))
    return affine_transform
    


def rescale_affine(small_affine_path: str, factor: float) -> SimpleITK.SimpleITK.Transform:
    with open(small_affine_path) as f:
        my_var = list(map(float, f.read().split()))
    # Modify translation vector
    new_val_1 = my_var[2] * factor
    new_val_2 = my_var[5] * factor

    affine_transform = sitk.AffineTransform(2)
    affine_transform.SetTranslation((new_val_1, new_val_2))
    affine_transform.SetMatrix((my_var[0], my_var[1], my_var[3], my_var[4]))
    return affine_transform


def rescale_warp(small_warp_path: str,
                 big_warp_path: str,
                 small_resolution: image_shape,
                 original_resolution: image_shape,
                 factor: float):
    warp_sitk = sitk.ReadImage(small_warp_path)
    warp = sitk.GetArrayFromImage(warp_sitk)
    mask = np.ones((small_resolution[0], small_resolution[1], 2), dtype=np.uint8)
    padding = (warp.shape[0] - small_resolution[0])//2
    mask = np.pad(mask, ((padding, padding), (padding, padding), (0, 0)))
    warp_no_pad = (warp)[padding:-padding, padding:-padding]
    big_warp = resize(warp_no_pad, (original_resolution[0], original_resolution[1])) * factor
    big_warp_sitk = sitk.GetImageFromArray(big_warp, isVector=True)
    big_warp_sitk.SetDirection(warp_sitk.GetDirection())
    big_warp_sitk.SetOrigin((0,0))
    sitk.WriteImage(big_warp_sitk, big_warp_path)


def apply_mask(image: numpy.array, mask: numpy.array) -> numpy.array:
    if len(image.shape) == 2:
        return image * mask
    return image * np.expand_dims(mask, -1).astype(np.uint8)


def get_symmetric_padding(img1: numpy.array, img2: numpy.array) -> Tuple[padding_type, padding_type]:
    max_size = max(img1.shape[0], img1.shape[1], img2.shape[0], img2.shape[1])
    # print(max_size)
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
    if len(image.shape) == 3:
        image = rgb2gray(image)
    image = image * 255
    image = image.astype(np.float32)
    return image


def pad_image(image: numpy.array, padding: padding_type, constant_values: float = 0) -> numpy.array:
    dims = len(image.shape)
    if dims == 2:
        padded_image = np.pad(image, ((padding, padding), (padding, padding)), constant_values=constant_values)
    elif dims == 3:
        padded_image = np.pad(image, ((padding, padding), (padding, padding), (0,0)), constant_values=constant_values)
    else:
        pass # Throw error instead
    return padded_image


def pad_asym(image: numpy.array, padding: padding_type, constant_values: int = 0) -> numpy.array:
    left, right, top, bottom = padding
    if len(image.shape) == 2:
        image = np.pad(image, ((top, bottom), (left, right)), constant_values=constant_values)
    else:
        # Assume 3 dimensions
        image = np.pad(image, ((top, bottom), (left, right), (0, 0)), constant_values=constant_values)
    return image


def cropping(mask: numpy.array) -> Tuple[numpy.array, padding_type]:
    p = np.argwhere(mask == 1)
    min_x = int(np.min(p[:,0]))
    max_x = int(np.max(p[:,0]))
    min_y = int(np.min(p[:,1]))
    max_y = int(np.max(p[:,1]))
    cropped_mask = mask[min_x:max_x, min_y:max_y]
    return cropped_mask, (min_x, max_x, min_y, max_y)    


def resample_by_factor(img: numpy.array, factor: float) -> numpy.array:
    h, w = img.shape[:2]
    img2 = cv2.resize(img, (int(w*factor), int(h*factor)))
    return img2


# TODO: Fix type for interpolator
def resample_image_sitk(image: numpy.array, 
                        scaling_factor: float, 
                        ref_image_shape: Optional[Tuple[int, int]] = None,
                        interpolator: int = sitk.sitkLinear):
    if scaling_factor == 1:
        return image
    if ref_image_shape is None:
        shape = image.shape[:2]
        ref_image_shape = (int(shape[0]*scaling_factor), int(shape[1]*scaling_factor))
    transform = sitk.ScaleTransform(2, (1/scaling_factor, 1/scaling_factor))
    ref_img = sitk.GetImageFromArray(np.zeros(ref_image_shape))
    
    sitk_image = sitk.GetImageFromArray(image, True)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref_img)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(transform)
    warped_image_sitk = resampler.Execute(sitk_image)
    resampled_image_np = sitk.GetArrayFromImage(warped_image_sitk)
    return resampled_image_np
    
    
def read_image(fpath: str, squeeze=False) -> numpy.array:
    if fpath.endswith('tiff') or fpath.endswith('tif'):
        image = tifffile.imread(fpath)
        return image
    sitk_image = sitk.ReadImage(fpath)
    image = sitk.GetArrayFromImage(sitk_image)
    if squeeze:
        image = np.squeeze(image)
    return image


def derive_resampling_factor(image: numpy.array, 
                             max_resample_dim = 3500) -> float:
    """Determines the resampling factor necessary for rescaling the 
    given image to max_resample_dim. If the image's maximum 
    resolution is smaller than max_resample_dim, a neutral 
    scaling factor of 1 will be returned.

    Args:
        image (numpy.array): 
        max_resample_dim (int, optional): Defaults to 3500.

    Returns:
        float: resample factor
    """
    max_dim = max(image.shape[0], image.shape[1])
    if max_dim <= max_resample_dim:
        return 1
    resample_factor = max_resample_dim / max_dim
    return resample_factor

# TODO: Set return type to displacement field
def realign_displacement_field(path: str) -> SimpleITK.SimpleITK.CompositeTransform:
    """Reads and rotates the rotation field from "NIFTI" to "DICOM" orientation. Needed because Greedy outputs transforms in NIFTI orientation.

    Args:
        path (str):

    Returns:
        SimpleITK.SimpleITK.CompositeTransform:
    """
    displacement_field = sitk.ReadImage(path, sitk.sitkVectorFloat64)
    rotated_displ_field = sitk.GetArrayFromImage(displacement_field)
    rotated_displ_field *= -1
    rotated_displ_field_sitk = sitk.GetImageFromArray(rotated_displ_field, True)
    displ_field = sitk.Image(rotated_displ_field_sitk) 
    displ_field = sitk.Cast(displ_field, sitk.sitkVectorFloat64)
    deformable_transform = sitk.DisplacementFieldTransform(2)
    deformable_transform.SetDisplacementField(displ_field)
    return deformable_transform


def translation_length(x: float, y: float) -> float:
    """Computes length of translation offset

    Args:
        x (float):
        y (float):

    Returns:
        float:
    """
    
    return np.sqrt(np.square(x) + np.square(y))