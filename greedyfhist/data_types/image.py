from dataclasses import dataclass
import os
from os.path import join, exists
from typing import Any

import numpy
import numpy as np
import SimpleITK as sitk

from greedyfhist.registration.greedy_f_hist import GreedyFHist, RegistrationResult
from greedyfhist.utils.io import derive_output_path

@dataclass
class DefaultImage:

    data: numpy.array
    path: str
    is_annotation: bool = False
    keep_axis: bool = False
    
    def to_file(self, path):
        if self.keep_axis:
            img = np.moveaxis(self.data, 2, 0)
        else:
            img = self.data
        sitk.WriteImage(sitk.GetImageFromArray(img), path)

    def to_directory(self, directory: str):
        fname = os.path.basename(self.path)
        output_path = derive_output_path(directory, fname)
        self.to_file(output_path)

        
    def transform_data(self, registerer: GreedyFHist, transformation: RegistrationResult) -> 'DefaultImage':
        interpolation = 'LINEAR' if not self.is_annotation else 'NN'
        warped_data = registerer.transform_image(self.data, transformation.forward_transform, interpolation)
        return DefaultImage(
            data=warped_data,
            path=self.path,
            is_annotation=self.is_annotation,
            keep_axis=self.keep_axis
        )

    # @staticmethod
    # def transform_data(image: 'DefaultImage', registerer: GreedyFHist, transformation: RegistrationResult) -> 'DefaultImage':
    #     interpolation = 'LINEAR' if not image.is_annotation else 'NN'
    #     warped_data = registerer.transform_image(image.data, transformation.forward_transform, interpolation)
    #     return DefaultImage(
    #         data=warped_data,
    #         is_annotation=image.is_annotation,
    #         switch_axis=image.switch_axis
    #     )
    
    @staticmethod
    def load_and_transform_data(path: str, 
                                registerer: GreedyFHist,
                                transformation: RegistrationResult,
                                keep_axis: bool = True,
                                is_annotation: bool = False):
        image = DefaultImage.load_from_path(path, keep_axis=keep_axis, is_annotation=is_annotation)
        warped_image = image.transform_data(registerer, transformation)
        return warped_image

    @classmethod
    def load_data(cls, dct):
        path = dct['path']
        switch_axis = dct.get('keep_axis', True)
        is_annotation = dct.get('is_annotation', False)
        data = sitk.GetArrayFromImage(sitk.ReadImage(path))
        if switch_axis:
            data = np.moveaxis(data, 0, 2)
        return cls(data, path, is_annotation, switch_axis)
    
    @classmethod
    def load_from_path(cls, path: str, keep_axis: bool = True, is_annotation: bool = False) -> 'DefaultImage':
        data = sitk.GetArrayFromImage(sitk.ReadImage(path))
        if is_annotation and not keep_axis:
            data = np.moveaxis(data, 0, 2)
        print('default image', data.shape)
        return cls(data, path, is_annotation, keep_axis)