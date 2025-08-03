"""
This module contains various dataclasses used for internal and external processing of registration results.
"""


from dataclasses import dataclass
import json
import os
from os.path import join, exists
import subprocess
from typing import Any, Optional

import SimpleITK, SimpleITK as sitk

from greedyfhist.utils.utils import composite_sitk_transforms
from greedyfhist.utils.io import create_if_not_exists


@dataclass
class GroupwiseRegResult:
    """Collection of all transforms computed during groupwise registration.

    Attributes
    ----------
    affine_transform: List[GFHTransform]
        List of affine transforms. Affine transforms contain transform from current index of the transform in the list to the next index. 
        Order of affine transforms is based on the order images supplied to affine registration.

    reverse_affine_transform: List[GFHTransform]

    nonrigid_transform: List[GFHTransform]
        List of nonrigid transforms. Each transform warps from an affine transformed image to the fixed image.

    reverse_nonrigid_transform: List[GFHTransform]

    Methods
    -------
        
    get_transforms(source): int
        Computes the end-to-end transformation from source image to fixed image.

    """

    affine_transform: list['RegistrationTransforms']
    reverse_affine_transform: list['RegistrationTransforms']
    nonrigid_transform: list['RegistrationTransforms']
    reverse_nonrigid_transform: list['RegistrationTransforms']

    def __get_transform_to_fixed_image(self,
                                       source:int) -> 'RegistrationResult':
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
                       target: int | None = None,
                       skip_nonrigid: bool = False) -> 'RegistrationResult':
        """Retrieves registration from one moving image indexed by 'source'.

        Args:
            source (int): Index of moving image.
            target (int): Index of reference image.
            skip_nonrigid (bool): If True, skip registration from nonrigid transformation.

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
    def __save_transforms_to_file(transform_list: list['RegistrationTransforms'], path: str):
        create_if_not_exists(path)
        for idx, transform in enumerate(transform_list):
            dir_i = join(path, f'{idx}')
            create_if_not_exists(dir_i)
            transform.to_directory(dir_i)

    @staticmethod
    def __load_transforms_from_dir(path: str) -> list['RegistrationTransforms']:
        sub_dirs = os.listdir(path)
        sub_dirs = sorted(sub_dirs, lambda x: int(sub_dirs)) # type: ignore
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

    transform: SimpleITK.Transform
        Transform from source to target image space.

    Methods
    -------

    to_file(path): str
        Saves GFHTransform to file.

    load_transform(path): str
        Loads transform from file.
    """
    
    size: tuple[int, int] | tuple[int, int, int]
    transform: SimpleITK.Transform

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
        self.transform.FlattenTransform() # type: ignore
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
    """Contains the result of a registration task (e.g. affine, or nonrigid). 
    The transformation from moving to fixed image is stored in the `registration` attribute.
    `reverse_registration` can be, if computed, used to transform from fixed to moving image space.
    Not needed in normal cases (might be used in groupwise registration).
    
    Attributes
    ----------
    
    registration: RegistrationTransforms
        Transformation from moving to fixed image space.
        
    reverse_registration: Optional[RegistrationTransforms]
        Transformation from fixed to moving image space.
        
    Methods
    -------
    
    to_directory(path): str
        Saves `RegistrationResult` to file.
    
    load(path): str
        Loads a `RegistrationResult` from file.
    """

    registration: 'RegistrationTransforms'

    reverse_registration: 'RegistrationTransforms | None' = None

    reg_params: Any | None = None

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
        """Loads a saved RegistrationResult directory.

        Args:
            path (str): Source path

        Returns:
            RegistrationResult: 
        """
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
    cmdln_returns: list[subprocess.CompletedProcess] | None = None
    reg_params: 'dict | list | InternalRegParams | None' = None
    
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
    cmdl_log: list[subprocess.CompletedProcess] | None
    reg_params: Any | None
    moving_preprocessing_params: dict
    fixed_preprocessing_params: dict
    path_to_small_ref_image: str
    sub_dir_key: int
    displacement_field: SimpleITK.Image | None
    inv_displacement_field: SimpleITK.Image | None


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
    
    
def compose_transforms(gfh_transforms: list['GFHTransform']) -> 'GFHTransform':
    """Composes a list of gfh_transforms.

    Args:
        gfh_transforms (List[GFHTransform]):

    Returns:
        GFHTransform:
    """
    composited_transform = composite_sitk_transforms([x.transform for x in gfh_transforms])
    gfh_comp_trans = GFHTransform(gfh_transforms[-1].size, composited_transform)
    return gfh_comp_trans    


# TODO: Check types here.
def compose_registration_results(reg_results: list['RegistrationResult']) -> 'RegistrationResult':
    """Composites all registrations in a list of RegistrationResults into one RegistrationResult.
    Target size are taken from the last RegistrationResult in the list.

    Args:
        reg_results (list[RegistrationResult]):

    Returns:
        RegistrationResult:
    """
    reg_fw = compose_transforms([x.registration.forward_transform for x in reg_results])
    reg_bw = compose_transforms([x.registration.backward_transform for x in reg_results[::-1]])
    rev_reg_fw = compose_transforms([x.reverse_registration.forward_transform for x in reg_results if x.reverse_registration])
    rev_reg_bw = compose_transforms([x.reverse_registration.backward_transform for x in reg_results[::-1] if x.reverse_registration])

    reg_transforms = RegistrationTransforms(forward_transform=reg_fw,
                                            backward_transform=reg_bw)
    rev_reg_transforms = RegistrationTransforms(forward_transform=rev_reg_fw,
                                            backward_transform=rev_reg_bw)

    reg_result = RegistrationResult(registration=reg_transforms, reverse_registration=rev_reg_transforms)
    return reg_result