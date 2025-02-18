"""
Utils for transformation of image and warp functions.
"""
import cv2
import numpy, numpy as np
import scipy.ndimage as nd
import SimpleITK, SimpleITK as sitk
from skimage.transform import resize
from skimage.filters import gaussian
from skimage.color import rgb2gray

from greedyfhist.custom_types import padding_type, image_shape


def com_affine_matrix(fixed: numpy.ndarray, moving: numpy.ndarray) -> numpy.ndarray:
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


def get_com_offset(mat: numpy.ndarray) -> float:
    """
    Computes the translation offset of the given matrix.
    
    Args:
        mat (numpy.ndarray):
            transform matrix
            
    Returns:
        translation offset
    """
    translation = mat[:2, 2]
    return np.sqrt(np.square(translation[0]) + np.square(translation[1]))


def read_affine_transform(small_affine_path: str) -> SimpleITK.AffineTransform:
    """Reads affine transform from file.

    Args:
        small_affine_path (str): Source file path.

    Returns:
        SimpleITK.AffineTransform:
    """
    with open(small_affine_path) as f:
        my_var = list(map(float, f.read().split()))
    # Modify translation vector

    affine_transform = sitk.AffineTransform(2)
    affine_transform.SetTranslation((my_var[2], my_var[5]))
    affine_transform.SetMatrix((my_var[0], my_var[1], my_var[3], my_var[4]))
    return affine_transform
    

def rescale_affine(small_affine_path: str, factor: float) -> SimpleITK.AffineTransform:
    """Reads affine transformation matrix and scales it by factor.

    Args:
        small_affine_path (str): 
        factor (float): 

    Returns:
        SimpleITK.AffineTransform:
    """
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
    """Reads a displacement field, removes padding, scales it by factor and writes it to a new file.

    Args:
        small_warp_path (str): Source file path.
        big_warp_path (str): Target file path.
        small_resolution (image_shape): Source resolution without padding
        original_resolution (image_shape): Target resolution
        factor (float): Scaling factor.
    """
    warp_sitk = sitk.ReadImage(small_warp_path)
    warp = sitk.GetArrayFromImage(warp_sitk)
    padding = (warp.shape[0] - small_resolution[0])//2
    warp_no_pad = warp[padding:-padding, padding:-padding]
    big_warp = resize(warp_no_pad, (original_resolution[0], original_resolution[1])) * factor
    big_warp_sitk = sitk.GetImageFromArray(big_warp, isVector=True)
    big_warp_sitk.SetDirection(warp_sitk.GetDirection())
    big_warp_sitk.SetOrigin((0,0))
    sitk.WriteImage(big_warp_sitk, big_warp_path)


def apply_mask(image: numpy.ndarray, mask: numpy.ndarray) -> numpy.ndarray:
    """Applies mask to image.

    Args:
        image (numpy.ndarray): 
        mask (numpy.ndarray): 

    Returns:
        numpy.ndarray: 
    """
    if len(image.shape) == 2:
        return image * mask
    return image * np.expand_dims(mask, -1).astype(np.uint8)


def get_symmetric_padding(img1: numpy.ndarray, img2: numpy.ndarray) -> tuple[padding_type, padding_type]:
    """Get padding parameters to make img1 and img2 uniform.

    Args:
        img1 (numpy.ndarray): 
        img2 (numpy.ndarray): 

    Returns:
        tuple[padding_type, padding_type]: 
    """
    max_size = max(img1.shape[0], img1.shape[1], img2.shape[0], img2.shape[1])
    # print(max_size)
    padding_img1 = get_padding_params(img1, max_size)
    padding_img2 = get_padding_params(img2, max_size)
    return padding_img1, padding_img2


def get_padding_params(img: numpy.ndarray, shape: int) -> padding_type:
    """Get padding parameters for img.

    Args:
        img (numpy.ndarray):
        shape (int): Square shape.

    Returns:
        padding_type: Padding
    """
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


def denoise_image(image: numpy.ndarray, 
                  resolution: int = 512, 
                  sp: int = 20, 
                  sr: int = 20, 
                  maxLevel: int = 2) -> numpy.ndarray:
    """Applies mean shift filtering to denoise the input image:
        - Image is changed from RGB to HSV.
        - Downscaling
        - Mean shift filtering.
        - Upscaling. 

    Args:
        image (numpy.ndarray): Input image.
        resolution (int, optional): Downscaling applied prior to denoising. Defaults to 512.
        sp (int, optional): Spatial window radius. Defaults to 20.
        sr (int, optional): Color window radius. Defaults to 20.
        maxLevel (int, optional): Maximum recursion level. Defaults to 2.

    Returns:
        numpy.ndarray: Denoised input image.
    """
    shape = image.shape
    img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    img_hsv = cv2.resize(img_hsv, (resolution, resolution))
    img_denoised = cv2.pyrMeanShiftFiltering(img_hsv, sp, sr, maxLevel)
    img_denoised = cv2.cvtColor(img_denoised, cv2.COLOR_HSV2RGB)
    img_denoised = cv2.resize(img_denoised, (shape[1], shape[0]))
    return img_denoised


def resample_image_with_gaussian(image: numpy.ndarray, resolution: image_shape, sigma: float) -> numpy.ndarray:
    """Resamples image to target resolution. Gaussian smoothing is applied beforehand to 
    help with antialiasing effects.

    Args:
        image (numpy.ndarray): 
        resolution (image_shape): 
        sigma (float): Applied to gaussian.

    Returns:
        numpy.ndarray: Resampled image.
    """
    image = gaussian(image, sigma, channel_axis=-1)
    image = resize(image, resolution)
    if len(image.shape) == 3:
        image = rgb2gray(image)
    image = image * 255
    image = image.astype(np.float32)
    return image


def pad_image(image: numpy.ndarray, padding: int, constant_values: float = 0) -> numpy.ndarray:
    """Pads image symmetrically with given padding value.

    Args:
        image (numpy.ndarray): 
        padding (padding_type): 
        constant_values (float, optional): Padding value. Defaults to 0.

    Returns:
        numpy.ndarray: Padded image.
    """
    dims = len(image.shape)
    if dims == 2:
        padded_image = np.pad(image, ((padding, padding), (padding, padding)), constant_values=constant_values)
    elif dims == 3:
        padded_image = np.pad(image, ((padding, padding), (padding, padding), (0,0)), constant_values=constant_values)
    else:
        raise Exception(f'Cannot handle image padding for dimension {dims}.')
    return padded_image


def pad_asym(image: numpy.ndarray, padding: padding_type, constant_values: int = 0) -> numpy.ndarray:
    """Pads image with given padding information.

    Args:
        image (numpy.ndarray): 
        padding (padding_type): 
        constant_values (float, optional): Padding value. Defaults to 0.

    Returns:
        numpy.ndarray: Padded image.
    """    
    left, right, top, bottom = padding
    if len(image.shape) == 2:
        image = np.pad(image, ((top, bottom), (left, right)), constant_values=constant_values)
    else:
        # Assume 3 dimensions
        image = np.pad(image, ((top, bottom), (left, right), (0, 0)), constant_values=constant_values)
    return image


def cropping(mask: numpy.ndarray, crop_value: int = 1) -> tuple[numpy.ndarray, padding_type]:
    """Crops image around given value.

    Args:
        mask (numpy.ndarray): 
        crop_value (int, optional): Defaults to 1.

    Returns:
        tuple[numpy.ndarray, padding_type]: Cropped image and cropping parameters.
    """
    p = np.argwhere(mask == crop_value)
    min_x = int(np.min(p[:,0]))
    max_x = int(np.max(p[:,0]))
    min_y = int(np.min(p[:,1]))
    max_y = int(np.max(p[:,1]))
    cropped_mask = mask[min_x:max_x, min_y:max_y]
    return cropped_mask, (min_x, max_x, min_y, max_y)    


def resample_by_factor(img: numpy.ndarray, factor: float) -> numpy.ndarray:
    """Resample image by a given factor.

    Args:
        img (numpy.ndarray): 
        factor (float):

    Returns:
        numpy.ndarray: Resampled image.
    """
    h, w = img.shape[:2]
    img2 = cv2.resize(img, (int(w*factor), int(h*factor)))
    return img2


def resample_image_sitk(image: numpy.ndarray,
                        scaling_factor: float, 
                        ref_image_shape: tuple[int, int] | None = None,
                        interpolator: int = sitk.sitkLinear) -> numpy.ndarray:
    """Resample an image by a given factor using SimpleITK functionality.

    Args:
        image (numpy.ndarray): 
        scaling_factor (float): 
        ref_image_shape (tuple[int, int] | None, optional): Defaults to None.
        interpolator (int, optional): _description_. Defaults to sitk.sitkLinear.

    Returns:
        numpy.ndarray: Resampled image.
    """
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
    
    
def derive_resampling_factor(image: numpy.ndarray, 
                             max_resample_dim = 3500) -> float:
    """Determines the resampling factor necessary for rescaling the 
    given image to max_resample_dim. If the image's maximum 
    resolution is smaller than max_resample_dim, a neutral 
    scaling factor of 1 will be returned.

    Args:
        image (numpy.ndarray): 
        max_resample_dim (int, optional): Defaults to 3500.

    Returns:
        float: resample factor
    """
    max_dim = max(image.shape[0], image.shape[1])
    if max_dim <= max_resample_dim:
        return 1
    resample_factor = max_resample_dim / max_dim
    return resample_factor


def realign_displacement_field(path: str) -> SimpleITK.DisplacementFieldTransform:
    """Reads and rotates the rotation field from "NIFTI" to "DICOM" orientation. Needed because Greedy outputs transforms in NIFTI orientation.

    Args:
        path (str):

    Returns:
        SimpleITK.CompositeTransform:
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


def pad_image_square(img: numpy.ndarray) -> numpy.ndarray:
    """Returns an image with square shape. Ends of image are padded symmetrically.

    Args:
        img (numpy.ndarray): 

    Returns:
        numpy.ndarray: 
    """
    max_dim = np.max(img.shape[:2])
    padding = get_padding_params(img, max_dim)
    return pad_asym(img, padding)


def get_corner_pixels(image: numpy.ndarray, x_range: int, y_range: int) -> numpy.ndarray:
    """Get pixels from the image corners.

    Args:
        image (numpy.ndarray):
        x_range (int): range in x-axis
        y_range (int): range in y-axis

    Returns:
        numpy.ndarray: Flattened array of pixels
    """
    c_dl = image[:x_range, :y_range].ravel()
    c_dr = image[image.shape[0] - x_range: image.shape[0], :y_range].ravel()
    c_ul = image[:x_range, image.shape[1] - y_range: image.shape[1]].ravel()
    c_ur = image[image.shape[0] - x_range: image.shape[0], image.shape[1] - y_range: image.shape[1]].ravel()
    pxs_arr = np.hstack((c_dl, c_dr, c_ul, c_ur))
    return pxs_arr


def scale_image_to_max_dim(img: numpy.ndarray, target_resolution: int = 640) -> numpy.ndarray:
    """Scales an image such that its largest dimension corresponds to target_resolution.

    Args:
        img (numpy.ndarray): 
        target_resolution (int, optional): Defaults to 640.

    Returns:
        numpy.ndarray: 
    """
    max_res = max(img.shape[0], img.shape[1])
    factor = target_resolution / max_res
    w, h = img.shape[:2]
    wn, hn = int(w*factor), int(h*factor)
    return cv2.resize(img, (hn, wn)) 