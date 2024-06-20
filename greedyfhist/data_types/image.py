from dataclasses import dataclass
from typing import Any

import numpy
import numpy as np
import SimpleITK as sitk

from greedyfhist.registration.greedy_f_hist import GreedyFHist, RegistrationResult

@dataclass
class DefaultImage:

    data: numpy.array
    is_annotation: bool = False
    switch_axis: bool = False
    
    def to_file(self, path):
        if self.switch_axis:
            img = np.moveaxis(self.data, 2, 0)
        else:
            img = self.data
        sitk.WriteImage(sitk.GetImageFromArray(img), path)
    
    def transform_data_method(self, registerer: GreedyFHist, transformation: RegistrationResult) -> 'DefaultImage':
        interpolation = 'LINEAR' if not self.is_annotation else 'NN'
        warped_data = registerer.transform_image(self.data, transformation.forward_transform, interpolation)
        return DefaultImage(
            data=warped_data,
            is_annotation=self.is_annotation,
            switch_axis=self.switch_axis
        )

    @staticmethod
    def transform_data(image: 'DefaultImage', registerer: GreedyFHist, transformation: RegistrationResult) -> 'DefaultImage':
        interpolation = 'LINEAR' if not image.is_annotation else 'NN'
        warped_data = registerer.transform_image(image.data, transformation.forward_transform, interpolation)
        return DefaultImage(
            data=warped_data,
            is_annotation=image.is_annotation,
            switch_axis=image.switch_axis
        )
    
    @staticmethod
    def load_and_transform_data(path: str, 
                                registerer: GreedyFHist,
                                transformation: RegistrationResult,
                                switch_axis: bool = False,
                                is_annotation: bool = False):
        image = DefaultImage.load_from_path(path, switch_axis=switch_axis, is_annotation=is_annotation)
        warped_image = DefaultImage.transform_data(image, registerer, transformation)
        return warped_image

    @classmethod
    def load_from_config(cls, dct):
        path = dct['path']
        switch_axis = dct.get('switch_axis', False)
        is_annotation = dct.get('is_annotation', False)
        img = sitk.GetArrayFromImage(sitk.ReadImage(path))
        if switch_axis:
            img = np.moveaxis(img, 0, 2)
        return cls(img, is_annotation, switch_axis)
    
    @classmethod
    def load_from_path(cls, path: str, switch_axis: bool = False, is_annotation: bool = False) -> 'DefaultImage':
        data = sitk.GetArrayFromImage(sitk.ReadImage(path))
        if switch_axis:
            data = np.moveaxis(data, 0, 2)
        return cls(data, is_annotation, switch_axis)