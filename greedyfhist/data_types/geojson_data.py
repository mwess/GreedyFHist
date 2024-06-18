from dataclasses import dataclass
from typing import Dict, List, Union

import geojson

from greedyfhist.registration.greedy_f_hist import GreedyFHist, RegistrationResult

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
        with open(path, 'wb') as f:
            geojson.dump(self.data, f)

    @staticmethod
    def transform_data(geojson_data: 'GeoJsonData', registerer: GreedyFHist, transformation: RegistrationResult) -> 'GeoJsonData':
        data = geojson_data.data.copy()
        warped_data = registerer.transform_geojson(data, transformation.backward_transform)
        return GeoJsonData(warped_data, geojson_data.path)
    
    @staticmethod
    def load_and_transform_data(path: str, 
                                registerer: GreedyFHist,
                                transformation: RegistrationResult,
                                switch_axis: bool = False,
                                is_annotation: bool = False):
        geojson_data = GeoJsonData.load_from_path(path)
        warped_geojson_data = GeoJsonData.transform_data(geojson_data, registerer, transformation)
        return warped_geojson_data
    
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
        
