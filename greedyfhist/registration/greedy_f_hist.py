"""GreedyFHist, the registration algorithm, includes pairwise/groupwise registration, transform for various file formats. 

This module handles the core registration functionality via the 
GreedyFHist class. Results are exported as GroupwiseRegResult or 
RegistrationTransforms. 
"""

from dataclasses import dataclass, field
import json
import os
from os.path import join, exists
import shutil
import subprocess
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import geojson
import numpy, numpy as np
import SimpleITK, SimpleITK as sitk

from greedyfhist.utils.io import create_if_not_exists, write_mat_to_file, clean_if_exists
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
from greedyfhist.utils.utils import deformable_registration, affine_registration, composite_warps, affine_registration, deformable_registration
from greedyfhist.segmentation import load_yolo_segmentation
from greedyfhist.options import RegistrationOptions, PreprocessingOptions


# TODO: Write serialization
@dataclass
class GroupwiseRegResult:
    """Collection of all transforms computed during groupwise registration.

    Attributes
    ----------
    affine_transform: List[GFHTransform]
        List of affine transforms. Affine transforms contain transform from current index of the transform in the list to the next index. 
        Order of affine transforms is based on the order images supplied to affine registration.
    
    deformable_transform: List[GFHTransform]
        List of nonrigid transforms. Each transform warps from an affinely transformed image to the fixed image.

    Methods
    -------
        
    get_transforms(source): int
        Computes the end-to-end transformation from source image to fixed image.

    """

    affine_transform: List['RegistrationTransforms']
    reverse_affine_transform: List['RegistrationTransforms']
    nonrigid_transform: List['RegistrationTransforms']
    reverse_nonrigid_transform: List['RegistrationTransforms']

    def __get_transform_to_fixed_image(self,
                                       source:int) -> 'RegistrationTransforms':
        """Retrieves registration from one moving image indexed by 'source'.

        Args:
            source (int): Index of moving image.

        Returns:
            RegistrationTransforms: transformation from source to reference image.
        """
        # TODO: At the moment only one direction works.
        transforms = self.affine_transform[source:]
        if self.nonrigid_transform is not None and len(self.nonrigid_transform) > 0:
            transforms.append(self.nonrigid_transform[source])
        # Composite transforms
        composited_forward_transform = compose_transforms([x.forward_transform for x in transforms])
        composited_backward_transform = compose_transforms([x.backward_transform for x in transforms][::-1])
        registration = RegistrationTransforms(composited_forward_transform, composited_backward_transform)
        reverse_registration = RegistrationTransforms(composited_backward_transform, composited_forward_transform)
        reg_result = RegistrationResult(registration, reverse_registration)
        return reg_result

    def get_transforms(self,
                       source: int,
                       target: Optional[int] = None,
                       skip_nonrigid: bool = False) -> 'RegistrationTransforms':
        """Retrieves registration from one moving image indexed by 'source'.

        Args:
            source (int): Index of moving image.

        Returns:
            RegistrationTransforms: transformation from source to reference image.
        """
        if target is None:
            return self.__get_transform_to_fixed_image(source)
        if source < target:
            reverse = False
        else:
            reverse = True

        if not reverse:
            transforms = [self.affine_transform[x] for x in range(source, target)]
        else:
            transforms = [self.reverse_affine_transform[x] for x in range(source - 1, target - 1, -1)]

        if not skip_nonrigid and self.nonrigid_transform is not None and len(self.nonrigid_transform) > 0:
            if not reverse:
                transforms.append(self.nonrigid_transform[target-1])
            else:
                transforms.append(self.reverse_nonrigid_transform[target])
            
        # Composite transforms
        composited_forward_transform = compose_transforms([x.forward_transform for x in transforms])
        composited_backward_transform = compose_transforms([x.backward_transform for x in transforms][::-1])
        registration = RegistrationTransforms(composited_forward_transform, composited_backward_transform)
        reverse_registration = RegistrationTransforms(composited_backward_transform, composited_forward_transform)
        reg_result = RegistrationResult(registration, reverse_registration)
        return reg_result
    
    def to_file(self, path: str):
        aff_dir = join(path, 'affine_transforms')
        GroupwiseRegResult.__save_transforms_to_file(self.affine_transform, aff_dir)
        aff_rev_dir = join(path, 'reverse_affine_transforms')
        GroupwiseRegResult.__save_transforms_to_file(self.reverse_affine_transform, aff_rev_dir)
        nr_dir = join(path, 'nonrigid_transforms')
        GroupwiseRegResult.__save_transforms_to_file(self.nonrigid_transform, nr_dir)
        nr_rev_dir = join(path, 'reverse_nonrigid_transforms')
        GroupwiseRegResult.__save_transforms_to_file(self.reverse_nonrigid_transform, nr_rev_dir)
    
    @staticmethod
    def __save_transforms_to_file(transform_list: List['RegistrationTransforms'], path: str):
        create_if_not_exists(path)
        for idx, transform in enumerate(transform_list):
            dir_i = join(path, f'{idx}')
            create_if_not_exists(dir_i)
            transform.to_directory(dir_i)

    @staticmethod
    def __load_transforms_from_dir(path: str) -> List['GroupwiseRegResult']:
        sub_dirs = os.listdir(path)
        sub_dirs = sorted(sub_dirs, lambda x: int(sub_dirs))
        transforms = []
        for sub_dir in sub_dirs:
            path = join(path, sub_dir)
            transform = RegistrationTransforms.load(path)
            transforms.append(transform)
        return transforms

    @staticmethod
    def load(path: str) -> 'GroupwiseRegResult':
        aff_dir = join(path, 'affine_transforms')
        aff_transforms  = GroupwiseRegResult.__load_transforms_from_dir(aff_dir)
        rev_aff_dir = join(path, 'reverse_affine_transforms')
        rev_aff_transforms = GroupwiseRegResult.__load_transforms_from_dir(rev_aff_dir)
        nr_dir = join(path, 'nonrigid_transforms')
        nr_transforms = GroupwiseRegResult.__load_transforms_from_dir(nr_dir)
        nr_rev_dir = join(path, 'reverse_nonrigid_transforms')
        rev_nr_transforms = GroupwiseRegResult.__load_transforms_from_dir(nr_rev_dir)
        return GroupwiseRegResult(
            affine_transform=aff_transforms,
            reverse_affine_transform=rev_aff_transforms,
            nonrigid_transform=nr_transforms,
            reverse_nonrigid_transform=rev_nr_transforms
        )


@dataclass
class GFHTransform:
    """
    Contains transform from one image space to another.

    Attributes
    ----------

    size: Tuple[int, int]
        Resolution of target image space.

    transform: SimpleITK.SimpleITK.Transform
        Transform from source to target image space.

    Methods
    -------

    to_file(path): str
        Saves GFHTransform to file.

    load_transform(path): str
        Loads transform from file.
    """
    
    size: Tuple[int, int]
    transform: SimpleITK.SimpleITK.Transform

    # TODO: Check that path is directory and change name since we are storing to a directory and not to one file.
    def to_file(self, path: str):
        """Saves transform to hard drive. Note, transforms are flattened before storing.

        Args:
            path (str): Location to store.
        """
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
    def load_transform(path: str) -> 'GFHTransform':
        """Load transform from directory.

        Args:
            path (str): Source location

        Returns:
            GFHTransform: 
        """
        attributes_path = join(path, 'attributes.json')
        with open(attributes_path) as f:
            attributes = json.load(f)
        size = (attributes['width'], attributes['height'])
        transform_path = join(path, 'transform.txt')
        transform = sitk.ReadTransform(transform_path)
        return GFHTransform(size, transform)


@dataclass
class RegistrationResult:

    registration: 'RegistrationTransforms'

    reverse_registration: Optional['RegistrationTransforms']

    def to_directory(self, path: str):
        """
        Save 'RegistrationResult' to file.
        """
        create_if_not_exists(path)
        reg_path = join(path, 'registration')
        self.registration.to_directory(reg_path)
        if self.reverse_registration is not None:
            rev_reg_path = join(path, 'reverse_registration')
            self.reverse_registration.to_directory(rev_reg_path)

    @staticmethod
    def load(path: str) -> 'RegistrationResult':
        reg_path = join(path, 'registration')
        registration = RegistrationTransforms.load(reg_path)
        rev_reg_path = join(path, 'reverse_registration')
        if exists(rev_reg_path):
            reverse_registration = RegistrationTransforms.load(rev_reg_path)
        else:
            reverse_registration = None
        return RegistrationResult(registration, reverse_registration)
        

@dataclass
class RegistrationTransforms:
    """
    Result of one pairwise registrations.

    Attributes
    ----------
    
    forward_transform: GFHTransform
        Transform from moving to fixed image space. Used for transforming image data from moving to fixed image space.

    backward_transform: GFHTransform
        Transform from fixed to moving image space. Used for transforming pointset data from moving to fixed image space.

    cmdln_returns: Optional[List[subprocess.CompletedProcess]]
        Contains log output from command line executions.

    reg_params: Dict
        Contains internally computed registration parameters.

    Methods
    -------

    to_file(path): str
        Saves RegistrationTransforms to file.

    load(path): str -> RegistrationTransforms
        Load RegistrationTransforms from file.
    """
    
    forward_transform: 'GFHTransform'
    backward_transform: 'GFHTransform'
    cmdln_returns: Optional[List[subprocess.CompletedProcess]] = None
    reg_params: Optional[Dict] = None
    
    # TODO: Can I add cmdln_returns and reg_params somehow
    def to_directory(self, path: str):
        """Saves 'RegistrationTransforms' to file.

        Args:
            path (str): Directory location.
        """
        create_if_not_exists(path)
        forward_transform_path = join(path, 'fixed_transform')
        self.forward_transform.to_file(forward_transform_path)
        backward_transform_path = join(path, 'moving_transform')
        self.backward_transform.to_file(backward_transform_path)

    @staticmethod
    def load(path: str) -> 'RegistrationTransforms':
        """Load RegistrationTransforms from location.

        Args:
            path (str): Directory.

        Returns:
            RegistrationTransforms: 
        """
        fixed_transform_path = join(path, 'fixed_transform')
        fixed_transform = GFHTransform.load_transform(fixed_transform_path)
        moving_transform_path = join(path, 'moving_transform')
        moving_transform = GFHTransform.load_transform(moving_transform_path)
        return RegistrationTransforms(fixed_transform, moving_transform)


@dataclass
class InternalRegParams:
    """
    Collected params with several filenames, logs and registration parameters. Used to move information around for post processing.

    Attributes
    ----------

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


    Methods
    -------

    from_directory(directory) -> InternalRegParams
        Load from directory.

    """
    path_to_small_fixed: str
    path_to_small_moving: str
    path_to_small_composite: str
    path_to_big_composite: str
    path_to_small_inv_composite: str
    path_to_big_inv_composite: str
    cmdl_log: Optional[List[subprocess.CompletedProcess]]
    reg_params: Optional[Any]
    moving_preprocessing_params: Dict
    fixed_preprocessing_params: Dict
    path_to_small_ref_image: str
    sub_dir_key: int
    displacement_field: SimpleITK.SimpleITK.Image
    inv_displacement_field: SimpleITK.SimpleITK.Image

    @classmethod
    def from_directory(cls, directory: str) -> 'InternalRegParams':
        """Load from directory.

        Args:
            directory (str): 

        Raises:
            Exception: Thrown if directory does not exist.

        Returns:
            InternalRegResult: 
        """
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
    """
    Information about preprocessed image.

    Attributes
    ----------

    image_path: str
        Path to preprocessed image.
    
    height: int

    width: int

    height_padded: int

    width_padded: int

    height_original: int

    width_original: int
    """
    image_path: str
    height: int
    width: int
    height_padded: int
    width_padded: int
    height_original: int
    width_original: int


# TODO: Write a sub function that just works with simpleitk transforms.
def composite_sitk_transforms(transforms: List[SimpleITK.SimpleITK.Transform]) -> SimpleITK.SimpleITK.Transform:
    """Composites all Transforms into one composite transform.

    Args:
        transforms (List[SimpleITK.SimpleITK.Transform]): 

    Returns:
        SimpleITK.SimpleITK.Transform: 
    """
    composited_transform = sitk.CompositeTransform(2)
    for transform in transforms:
        composited_transform.AddTransform(transform)
    return composited_transform
    

def compose_transforms(gfh_transforms: List['GFHTransform']) -> 'GFHTransform':
    """Composes a list of gfh_transforms.

    Args:
        gfh_transforms (List[GFHTransform]):

    Returns:
        GFHTransform:
    """
    composited_transform = composite_sitk_transforms([x.transform for x in gfh_transforms])
    gfh_comp_trans = GFHTransform(gfh_transforms[-1].size, composited_transform)
    return gfh_comp_trans

def compose_reg_transforms(transform: SimpleITK.SimpleITK.Transform, 
                           moving_preprocessing_params: Dict,
                           fixed_preprocessing_params: Dict) -> SimpleITK.SimpleITK.Transform:
    """Pre- and appends preprocessing steps from moving and fixed image as transforms to forward affine/nonrigid registration.  

    Args:
        transform (SimpleITK.SimpleITK.Transform): Computed affine/nonrigid registration
        internal_reg_params (InternalRegParams): Contains parameters of preprocessing steps.
        reverse (bool): Switches moving and fixed preprocessing params if True.

    Returns:
        SimpleITK.SimpleITK.Transform: Composited end-to-end registration.
    """
    moving_padding = moving_preprocessing_params['padding']
    moving_cropping = moving_preprocessing_params['cropping_params']
    fixed_padding = fixed_preprocessing_params['padding']
    fixed_cropping = fixed_preprocessing_params['cropping_params']
    mov_ds_factor = moving_preprocessing_params['resampling_factor']
    fix_ds_factor = fixed_preprocessing_params['resampling_factor']
    
    all_transforms = sitk.CompositeTransform(2)

    pre_downscale_transform = sitk.ScaleTransform(2, (1/mov_ds_factor, 1/mov_ds_factor))
    post_upscale_transform = sitk.ScaleTransform(2, (fix_ds_factor, fix_ds_factor))
    
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
                               moving_preprocessing_params: Dict,
                               fixed_preprocessing_params: Dict) -> SimpleITK.SimpleITK.Transform:
    """Pre- and appends preprocessing steps from moving and fixed image as transforms to backward affine/nonrigid registration.  

    Args:
        transform (SimpleITK.SimpleITK.Transform): Computed affine/nonrigid registration
        internal_reg_params (InternalRegParams): Contains parameters of preprocessing steps.

    Returns:
        SimpleITK.SimpleITK.Transform: Composited end-to-end transform.
    """
    moving_padding = moving_preprocessing_params['padding']
    moving_cropping = moving_preprocessing_params['cropping_params']
    fixed_padding = fixed_preprocessing_params['padding']
    fixed_cropping = fixed_preprocessing_params['cropping_params']
    mov_ds_factor = moving_preprocessing_params['resampling_factor']
    fix_ds_factor = fixed_preprocessing_params['resampling_factor']
    
    all_transforms = sitk.CompositeTransform(2)

    pre_downscale_transform = sitk.ScaleTransform(2, (1/fix_ds_factor, 1/fix_ds_factor))
    post_upscale_transform = sitk.ScaleTransform(2, (mov_ds_factor, mov_ds_factor))

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
    all_transforms.AddTransform(transform)
    all_transforms.AddTransform(aff_trans2)
    all_transforms.AddTransform(aff_trans1)
    all_transforms.AddTransform(post_upscale_transform)
    return all_transforms


def preprocessing(image: numpy.ndarray,
                  preprocessing_options: PreprocessingOptions,
                  resolution: Tuple[int, int],
                  kernel_size: int,
                  tmp_path: str,
                  skip_denoising: bool = False
                  ) -> 'PreprocessedData':
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


def affine_transform_to_file(transform: SimpleITK.SimpleITK.AffineTransform, fpath: str):
    mat = transform.GetMatrix()
    trans = transform.GetTranslation()
    mat_str = f'{mat[0]} {mat[1]} {trans[0]}\n{mat[2]} {mat[3]} {trans[1]}\n0 0 1'
    with open(fpath, 'w') as f:
        f.write(mat_str)


def preprocess_image_for_greedy(image: numpy.ndarray,
                     kernel: int,
                     resolution: Tuple[int, int],
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

        
def derive_subdir(directory: str, limit=1000) -> Tuple[str, int]:
    """Derives a unique subdirectory. Counts upwards until a new directory is found.

    Args:
        directory (_type_): 
        limit (int, optional): Maximum subdir count. Defaults to 1000.

    Returns:
        Tuple[str, int]: Subdir and final count.
    """
    for subdir_num in range(limit):
        subdir = f'{directory}/{subdir_num}'
        if not exists(subdir):
            return subdir, subdir_num
    # TODO: Do better error handling here, but who has 1000 sections to register, really?!
    return subdir, subdir_num


# TODO: There is probably a better way to ensure the dtype of the image.
def correct_img_dtype(img: numpy.ndarray) -> numpy.ndarray:
    """Changes the image type from float to np.uint8 if necessary.

    Args:
        img (numpy.ndarray): 

    Returns:
        numpy.ndarray: 
    """
    if np.issubdtype(img.dtype, np.floating):
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)
    return img


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
    segmentation_function: Optional[Callable] = None
    cmdln_returns: List[Any] = field(default_factory=lambda: [])

    def __post_init__(self):
        if self.segmentation_function is None:
            self.segmentation_function = load_yolo_segmentation()

    def register(self,
                 moving_img: numpy.ndarray,
                 fixed_img: numpy.ndarray,
                 moving_img_mask: Optional[numpy.ndarray] = None,
                 fixed_img_mask: Optional[numpy.ndarray] = None,
                 options: Optional[RegistrationOptions] = None) -> 'RegistrationResult':
        """Performs pairwise registration from moving_img to fixed_img. Optional tissue masks can be provided.
        Options are supplied via the options arguments.

        Documentation of the registration algorithm can be found here: ...
        

        Args:
            moving_img (numpy.ndarray): 
            fixed_img (numpy.ndarray): 
            moving_img_mask (Optional[numpy.ndarray], optional): Optional moving mask. Is otherwise derived automatically. Defaults to None.
            fixed_img_mask (Optional[numpy.ndarray], optional): Optional fixed mask. Is otherwise dervied automatically. Defaults to None.
            options (Optional[Options], optional): Can be supplied. Otherwise default arguments are used. Defaults to None.

        Returns:
            RegistrationResult: Contains computed registration result.
        """
        reg_result = self.register_(moving_img,
                                    fixed_img,
                                    moving_img_mask,
                                    fixed_img_mask,
                                    options)
        return reg_result 

    def register_(self,
                 moving_img: numpy.ndarray,
                 fixed_img: numpy.ndarray,
                 moving_img_mask: Optional[numpy.ndarray] = None,
                 fixed_img_mask: Optional[numpy.ndarray] = None,
                 options: Optional[RegistrationOptions] = None,                  
                 **kwargs: Dict) -> Tuple['RegistrationTransforms', Optional['RegistrationTransforms']]:
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
        pre_sampling_factor = options.pre_sampling_factor
        if pre_sampling_factor == 'auto':
            pre_sampling_max_img_size = options.pre_sampling_max_img_size
            if pre_sampling_max_img_size is not None:
                max_size = max(moving_img.shape[0], moving_img.shape[1], fixed_img.shape[0], fixed_img.shape[1])
                if max_size > pre_sampling_max_img_size:
                    resampling_factor = pre_sampling_max_img_size / max_size
                    moving_resampling_factor = resampling_factor
                    fixed_resampling_factor = resampling_factor
                else:
                    moving_resampling_factor = 1
                    fixed_resampling_factor = 1
            else:
                moving_resampling_factor = 1
                fixed_resampling_factor = 1
        else:
            moving_resampling_factor = 1
            fixed_resampling_factor = 1

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
                                   moving_img,
                                   fixed_img,
                                   moving_img_mask,
                                   fixed_img_mask,
                                   options,
                                   paths,
                                   reg_params,
                                   compute_reverse_nonrigid_registration=False):
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
                        moving_img_preprocessed,
                        fixed_img_preprocessed,
                        moving_preprocessing_params,
                        fixed_preprocessing_params,
                        options,
                        moving_img,
                        reg_params,
                        paths,
                        reverse=False):
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
                               image_mask_list: List[Union[Tuple[numpy.ndarray, Optional[numpy.ndarray]]]],
                               options: Optional[RegistrationOptions] = None,
                               ) -> Tuple[GroupwiseRegResult, List[numpy.ndarray]]:
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
        """Transforms image data from moving to fixed image space using computed transformation.

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
                        size: Tuple[int,int],
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
        """Transforms pointset from moving to fixed image space.

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

    def __warp_geojson_coord_tuple(self, coord: Tuple[float, float], transform: SimpleITK.SimpleITK.Transform) -> Tuple[float, float]:
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
    def load_from_config(cls, config: Optional[dict[str, Any]] = None) -> 'GreedyFHist':
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