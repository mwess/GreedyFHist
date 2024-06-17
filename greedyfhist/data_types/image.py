from dataclasses import dataclass
from typing import Any

import numpy as np
import SimpleITK as sitk

from greedyfhist.registration.greedy_f_hist import GreedyFHist, RegistrationResult

@dataclass
class Image:

    img: Any
    is_annotation: bool = False
    switch_axis: bool = False
    
    def update_data(self, registerer: GreedyFHist, transformation: RegistrationResult):
        interpolation = 'LINEAR' if not self.is_annotation else 'NN'
        warped_img = registerer.transform_image(self.img, transformation.forward_transform, interpolation)
        self.img = warped_img
    
    def to_file(self, path):
        if self.switch_axis:
            img = np.moveaxis(self.img, 2, 0)
        else:
            img = self.img
        sitk.WriteImage(sitk.GetImageFromArray(img), path)
    
    @classmethod
    def load_data(cls, dct):
        path = dct['path']
        switch_axis = dct.get('switch_axis', False)
        is_annotation = dct.get('is_annotation', False)
        img = sitk.GetArrayFromImage(sitk.ReadImage(path))
        if switch_axis:
            img = np.moveaxis(img, 0, 2)
        return cls(img, is_annotation, switch_axis)