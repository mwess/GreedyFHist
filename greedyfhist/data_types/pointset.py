from dataclasses import dataclass
from typing import Any, Optional

import numpy
import numpy as np

import pandas
import pandas as pd

from greedyfhist.registration.greedy_f_hist import GreedyFHist, RegistrationResult

@dataclass
class Pointset:
    
    data = Any  
    x_axis: str = 'x'
    y_axis: str = 'y'
    index_col: Optional[int] = None

    def to_numpy(self) -> numpy.array:
        return self.data[[self.x_axis, self.y_axis]].to_numpy()

    def update_data(self, registerer: GreedyFHist, transformation: RegistrationResult):

        pointset = self.to_numpy()
        warped_pointset = registerer.transform_pointset(pointset, transformation.moving_transform)
        
        self.data[self.x_axis] = warped_pointset[:, 0]
        self.data[self.y_axis] = warped_pointset[:, 1]

    def to_file(self, path):
        index = True if self.index_col is not None else False
        self.data.to_csv(path, index=index)

    @classmethod
    def load_data(cls, dct) -> 'Pointset':
        index_col = dct.get('index_col', None)
        path = dct['path']
        data = pd.read_csv(path, index_col=index_col)
        x_axis = dct.get('x', 'x')
        y_axis = dct.get('y', 'y')
        return cls(data, x_axis, y_axis, index_col)

