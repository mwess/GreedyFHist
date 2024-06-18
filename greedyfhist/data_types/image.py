from dataclasses import dataclass
from typing import Any

import numpy
import numpy as np
import SimpleITK as sitk

from greedyfhist.registration.greedy_f_hist import GreedyFHist, RegistrationResult

@dataclass
class Image:

    data: numpy.array
    is_annotation: bool = False
    switch_axis: bool = False
    
    def to_file(self, path):
        if self.switch_axis:
            img = np.moveaxis(self.data, 2, 0)
        else:
            img = self.data
        sitk.WriteImage(sitk.GetImageFromArray(img), path)
    
    @staticmethod
    def transform_data(image: 'Image', registerer: GreedyFHist, transformation: RegistrationResult) -> 'Image':
        interpolation = 'LINEAR' if not image.is_annotation else 'NN'
        warped_data = registerer.transform_image(image.data, transformation.forward_transform, interpolation)
        return Image(
            data=warped_data,
            is_annotation=image.is_annotation,
            switch_axis=image.switch_axis
        )
    
    @classmethod
    def load_data_from_config(cls, dct):
        path = dct['path']
        switch_axis = dct.get('switch_axis', False)
        is_annotation = dct.get('is_annotation', False)
        img = sitk.GetArrayFromImage(sitk.ReadImage(path))
        if switch_axis:
            img = np.moveaxis(img, 0, 2)
        return cls(img, is_annotation, switch_axis)
    
    @classmethod
    def load_data_from_path(cls, path: str, switch_axis: bool = False, is_annotation: bool = False) -> 'Image':
        data = sitk.GetArrayFromImage(sitk.ReadImage(path))
        if switch_axis:
            data = np.moveaxis(data, 0, 2)
        return cls(data, is_annotation, switch_axis)