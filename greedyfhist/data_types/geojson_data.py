from dataclasses import dataclass
import os
from os.path import join
from typing import Dict, List, Union

import geojson

from greedyfhist.registration.greedy_f_hist import GreedyFHist, RegistrationTransforms

@dataclass
class GeoJsonData:
    """
    Class for handling data in geojson format.
    
    Attributes
    ----------
    
    data: Union[geojson.feature.FeatureCollection, List[geojson.feature.Feature]]
        geojson data
        
    path: str
        original_file_path
    """
    
    data: Union[geojson.feature.FeatureCollection, List[geojson.feature.Feature]]
    path: str

    def to_file(self, path: str):
        with open(path, 'w') as f:
            geojson.dump(self.data, f)

    def to_directory(self, directory: str):
        fname = os.path.basename(self.path)
        path = join(directory, fname)
        self.to_file(path)

    def transform_data(self, registerer: GreedyFHist, transformation: RegistrationTransforms) -> 'GeoJsonData':
        data = self.data.copy()
        warped_data = registerer.transform_geojson(data, transformation.backward_transform)
        return GeoJsonData(warped_data, self.path)
    
    @classmethod
    def load_data(cls, dct: Dict) -> 'GeoJsonData':
        path = dct['path']
        with open(path, 'rb') as f:
            data = geojson.load(f)
        return cls(data, path)
    
    @classmethod
    def load_from_path(cls, path: str) -> 'GeoJsonData':
        with open(path, 'rb') as f:
            data = geojson.load(f)
        return cls(data, path)
        
