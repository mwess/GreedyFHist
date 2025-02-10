from dataclasses import dataclass
import os
from os.path import join

import numpy, numpy as np
import pandas, pandas as pd

from greedyfhist.registration.greedy_f_hist import GreedyFHist, RegistrationTransforms
from greedyfhist.utils.io import derive_output_path

@dataclass
class Pointset:
    """
    Class for handling pointset formatted data.
    
    Pointset data is stored as pandas DataFrames which allows to add 
    additional label and metainformation to each coordinate point.
    
    Attributes
    ----------
    
    data : pandas.DataFrame 
        Contains pointset valued data. 

    path: str
        Source path
        
    x_axis: str = 'x'
        Denotes column to access coordinates along the x axis.
        
    y_axis: str = 'y'
        Denotes column to access coordinates along the y axis.
        
    index_col: Optional[int] = None
        Denotes index column. Is useful to not introduce too
        many columns.

    header: Optional[int] = None
        Denotes header row. If none, no header is expected, and
        header columns 0 and 1 and renamed to x_axis and y_axis.
    """
    
    
    data : pandas.DataFrame 
    path: str
    x_axis: str = 'x'
    y_axis: str = 'y'
    index_col: int | None = None
    header: int | None = None

    def to_numpy(self) -> numpy.ndarray:
        return self.data[[self.x_axis, self.y_axis]].to_numpy()

    def to_file(self, path: str):
        index = True if self.index_col is not None else False
        header = True if self.header is not None else False
        self.data.to_csv(path, index=index, header=header)

    def to_directory(self, directory: str):
        fname = os.path.basename(self.path)
        output_path = join(directory, fname)
        self.to_file(output_path)

    def transform_data(self, registerer: GreedyFHist, transformation: RegistrationTransforms) -> 'Pointset':
        pointset_data = self.to_numpy()
        warped_pointset_data = registerer.transform_pointset(pointset_data, transformation.backward_transform)
        warped_data = self.data.copy()
        warped_data[self.x_axis] = warped_pointset_data[:, 0]
        warped_data[self.y_axis] = warped_pointset_data[:, 1]
        return Pointset(
            data=warped_data,
            path=self.path,
            x_axis=self.x_axis,
            y_axis=self.y_axis,
            index_col=self.index_col,
            header=self.header
        )

    @classmethod
    def load_data(cls, dct: dict) -> 'Pointset':
        index_col = dct.get('index_col', None)
        path = dct['path']
        x_axis = dct.get('x', 'x')
        y_axis = dct.get('y', 'y')
        header = dct.get('header', 0)
        data = pd.read_csv(path, index_col=index_col, header=header)
        if header is None:
            data.rename(columns={0: x_axis, 1: y_axis}, inplace=True)
        return cls(data, path, x_axis, y_axis, index_col, header)

    @classmethod
    def load_from_path(cls, path: str, x_axis='x', y_axis='y', index_col=None, header=0) -> 'Pointset':
        data = pd.read_csv(path, index_col=index_col, header=header)
        if header is None:
            data.rename(columns={0: x_axis, 1: y_axis}, inplace=True)
        return cls(data, path, x_axis, y_axis, index_col, header)