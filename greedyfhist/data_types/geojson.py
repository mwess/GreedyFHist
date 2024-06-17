from dataclasses import dataclass
from typing import List, Union

import geojson

from greedyfhist.registration.greedy_f_hist import GreedyFHist, RegistrationResult

@dataclass
class GeoJson:
    
    data = Union[geojson.feature.FeatureCollection, List[geojson.feature.Feature]]

    def to_file(self, path):
        with open(path, 'wb') as f:
            geojson.dump(self.data)

    def update_data(self, registerer: GreedyFHist, transformation: RegistrationResult):
        warped_data = registerer.transform_geojson(self.data, transformation)
        self.data = warped_data
    
    @classmethod
    def load_data(cls, dct) -> 'GeoJson':
        path = dct['path']
        with open(path, 'rb') as f:
            data = geojson.load(f)
        return cls(data)