from dataclasses import dataclass
from typing import Dict

import numpy
import numpy as np
from pyometiff import OMETIFFReader, OMETIFFWriter

from greedyfhist.registration.greedy_f_hist import GreedyFHist, RegistrationResult


# TODO: Fix this. The ome tiff written to file is kind of inefficient. Either switch to Tifffile or find out how to fix this with pyometiff.
@dataclass
class OMETIFFImage:
    """
    Class for processing OMETIFFImage. 
    
    
    Attributes
    ----------
    
    data: numpy.array
        Image data as a numpy array. 
        
    path: str
        Original file path.
        
    metadata: Dict
        OME TIFF metadata information. Necessary for writing back 
        to OME TIFF.
        
    is_annotation: bool
        If OME TIFF is an annotation this should be set to True, so 
        that Nearest Neighbor interpolation is used during 
        transformation.
        
    switch_axis: bool
        If set to True, first and third channel are switched when data 
        is loaded and switched back when writing data to file. 
        Normally, images are provided in W x H x C format, but some 
        tools generate images in format C x H x W, e.g. multi class 
        annotations from QuPath. 
    """
    
    data: numpy.array
    path: str
    metadata: Dict
    is_annotation: bool = False
    switch_axis: bool = False

    def to_file(self, path):
        metadata = self.metadata.copy()
        if self.switch_axis:
            img = np.moveaxis(self.data, 2, 0)
        else:
            img = self.data
        # explicit_tiffdata=False
        dimension_order = metadata['DimOrder BF Array']
        writer = OMETIFFWriter(
            fpath=path,
            dimension_order=dimension_order,
            array=img,
            metadata=metadata,
            explicit_tiffdata=False
        )
        writer.write()
    
    def transform_data_method(self, registerer: GreedyFHist, transformation: RegistrationResult) -> 'OMETIFFImage':
        interpolation = 'LINEAR' if not self.is_annotation else 'NN'
        warped_data = registerer.transform_image(self.data, transformation.forward_transform, interpolation)
        metadata = self.metadata.copy()
        metadata['SizeX'] = warped_data.shape[0]
        metadata['SizeY'] = warped_data.shape[1]        
        return OMETIFFImage(
            data=warped_data,
            path=self.path,
            metadata=metadata,
            is_annotation=self.is_annotation,
            switch_axis=self.switch_axis
        )

    @staticmethod
    def transform_data(image: 'OMETIFFImage', registerer: GreedyFHist, transformation: RegistrationResult):
        interpolation = 'LINEAR' if not image.is_annotation else 'NN'
        warped_data = registerer.transform_image(image.data, transformation.forward_transform, interpolation)
        metadata = image.metadata.copy()
        metadata['SizeX'] = warped_data.shape[0]
        metadata['SizeY'] = warped_data.shape[1]
        warped_image= OMETIFFImage(data=warped_data, 
                                   path=image.path,
                                   metadata=metadata, 
                                   is_annotation=image.is_annotation, 
                                   switch_axis=image.switch_axis)
        return warped_image
    
    @staticmethod
    def load_and_transform_data(path: str, 
                                registerer: GreedyFHist,
                                transformation: RegistrationResult,
                                switch_axis: bool = False,
                                is_annotation: bool = False):
        ome_tiff_image = OMETIFFImage.load_from_path(path, switch_axis=switch_axis, is_annotation=is_annotation)
        warped_ome_tiff_image = OMETIFFImage.transform_data(ome_tiff_image, registerer, transformation)
        return warped_ome_tiff_image
    
    @classmethod
    def load_from_config(cls, dct):
        path = dct['path']
        switch_axis = dct.get('switch_axis', False)
        is_annotation = dct.get('is_annotation', False)
        reader = OMETIFFReader(fpath=path)
        img, metadata, xml_metadata = reader.read()
        if switch_axis:
            img = np.moveaxis(img, 0, 2)
        return cls(img, path, metadata, is_annotation, switch_axis)
        
    @classmethod
    def load_from_path(cls, path, switch_axis=False, is_annotation=False):
        reader = OMETIFFReader(fpath=path)
        img, metadata, xml_metadata = reader.read()
        if switch_axis:
            img = np.moveaxis(img, 0, 2)
        return cls(img, path, metadata, is_annotation, switch_axis)
        
