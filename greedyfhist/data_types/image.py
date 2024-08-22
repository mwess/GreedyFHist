from dataclasses import dataclass
import os
from os.path import join
from typing import Dict, Optional

import numpy, numpy as np

from greedyfhist.registration.greedy_f_hist import GreedyFHist, RegistrationTransforms
from greedyfhist.utils.io import read_image, write_to_ometiffile


@dataclass
class Image:

    data: numpy.array
    path: str
    is_annotation: bool = False
    metadata: Optional[Dict] = None
    
    def transform_data(self, registerer: GreedyFHist, transformation: RegistrationTransforms) -> 'Image':
        interpolation = 'LINEAR' if not self.is_annotation else 'NN'
        warped_data = registerer.transform_image(self.data, transformation.forward_transform, interpolation)
        return Image(
            data=warped_data,
            path=self.path,
            is_annotation=self.is_annotation,
            metadata=self.metadata
        )

    def to_directory(self, directory: str):
        fname = os.path.basename(self.path)
        name, _ = os.path.splitext(fname)
        if name.endswith('.ome'):
            name = name.rsplit('.ome', maxsplit=1)[0]
        path = join(directory, f'{name}.ome.tif')
        self.to_file(path)

    def to_file(self, path: str):
        write_to_ometiffile(self.data, path, self.metadata, self.is_annotation)

    @classmethod
    def load_data_from_config(cls, dct):
        path = dct['path']
        is_annotation = dct.get('is_annotation', False)
        return Image.load_from_path(path, is_annotation)
    
    @classmethod
    def load_from_path(cls, path: str, is_annotation: bool = False) -> 'Image':
        data, metadata = read_image(path, is_annotation)
        if is_annotation:
            data = np.moveaxis(data, 0, 2)
        return cls(data, path, is_annotation, metadata)