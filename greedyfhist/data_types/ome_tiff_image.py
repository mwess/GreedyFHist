from dataclasses import dataclass
from typing import Dict

import numpy
import numpy as np
from pyometiff import OMETIFFReader, OMETIFFWriter

from greedyfhist.registration.greedy_f_hist import GreedyFHist, RegistrationResult


@dataclass
class OMETIFFImage:
    
    img: numpy
    metadata: Dict
    xml_metadta: str
    is_annotation: bool = False
    switch_axis: bool = False

    def update_data(self, registerer: GreedyFHist, transformation: RegistrationResult):
        interpolation = 'LINEAR' if not self.is_annotation else 'NN'
        warped_img = registerer.transform_image(self.img, transformation.fixed_transform, interpolation)
        self.img = warped_img
        self.metadata['SizeX'] = warped_img.shape[0]
        self.metadata['SizeY'] = warped_img.shape[1]

    def to_file(self, path):
        metadata = self.metadata.copy()
        if self.switch_axis:
            img = np.moveaxis(self.img, 2, 0)
        else:
            img = self.img
        writer = OMETIFFWriter(
            fpath=path,
            array=img,
            metadata=metadata,
            explicit_tiffdata=False
        )
        writer.write()
    
    @classmethod
    def load_data(cls, dct):
        path = dct['path']
        switch_axis = dct.get('switch_axis', False)
        is_annotation = dct.get('is_annotation', False)
        reader = OMETIFFReader(fpath=path)
        img, metadata, xml_metadata = reader.read()
        if switch_axis:
            img = np.moveaxis(img, 0, 2)
        return cls(img, metadata, xml_metadata, is_annotation, switch_axis)
        
