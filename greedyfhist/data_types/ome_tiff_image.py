from dataclasses import dataclass
from typing import Dict
import xml.etree.ElementTree as ET
import os

import numpy
import numpy as np
# from pyometiff import OMETIFFReader, OMETIFFWriter
import tifffile

from greedyfhist.registration.greedy_f_hist import GreedyFHist, RegistrationResult
from greedyfhist.utils.io import derive_output_path


@dataclass
class OMETIFFImage:
    """
    Class for processing TIFF and OMETIFF images. 
    
    
    Attributes
    ----------
    
    data: numpy.array
        Image data as a numpy array. 
        
    path: str
        Original file path.
        
    tif: tifffile.tifffile.TiffFile 
        Connection to tif image information.

    is_ome: bool
        Indicates whether the read file is ome.
        
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
    is_ome: bool
    tif: tifffile.tifffile.TiffFile
    is_annotation: bool = False
    switch_axis: bool = False
    

    def to_file(self, path: str):
        if self.is_ome:
            self.to_ome_tiff_file(path)
        else:
            self.to_tiff_file(path)

    def to_directory(self, directory: str):
        fname = os.path.basename(self.path)
        output_path = derive_output_path(directory, fname)
        self.to_file(output_path)

    def to_tiff_file(self, path: str):
        if self.switch_axis and len(self.data.shape) > 2:
            data = np.moveaxis(self.data, 2, 0)
        else:
            data = self.data
        tifffile.imwrite(path, data)        

    def to_ome_tiff_file(self, path: str):
        metadata = self.__get_metadata()
        if len(self.data.shape) == 2:
            options = {}
        else:
            options = dict(photometric='rgb')
        
        if self.switch_axis and len(self.data.shape) > 2:
            data = np.moveaxis(self.data, 2, 0)
        else:
            data = self.data

        with tifffile.TiffWriter(path, bigtiff=True) as tif:
            tif.write(
                data,
                metadata=metadata,
                **options
            )

    def __get_metadata(self) -> str:
        root = ET.fromstring(self.tif.ome_metadata)
        ns = {'ns0': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
        elem = root.findall('*/ns0:Pixels', ns)[0]
        metadata = elem.attrib
        # metadata['axes'] = 'TCYXS'
        del metadata['SizeX']
        del metadata['SizeY']
        del metadata['SizeC']
        del metadata['SizeT']
        del metadata['SizeZ']
        del metadata['DimensionOrder']
        return metadata
    
    def transform_data(self, registerer: GreedyFHist, transformation: RegistrationResult) -> 'OMETIFFImage':
        interpolation = 'LINEAR' if not self.is_annotation else 'NN'
        if not self.is_annotation or len(self.data.shape) < 3:
            warped_data = registerer.transform_image(self.data, transformation.forward_transform, interpolation)    
        else:
            warped_datas = []
            for channel_idx in range(self.data.shape[-1]):
                warped_layer = registerer.transform_image(self.data[:,:,channel_idx], transformation.forward_transform, interpolation)
                warped_datas.append(warped_layer)
            warped_data = np.dstack(warped_datas)
        return OMETIFFImage(
            data=warped_data,
            path=self.path,
            is_ome=self.is_ome,
            tif=self.tif,
            is_annotation=self.is_annotation,
            switch_axis=self.switch_axis
        )
    
    @staticmethod
    def load_and_transform_data(path: str, 
                                registerer: GreedyFHist,
                                transformation: RegistrationResult,
                                switch_axis: bool = False,
                                is_annotation: bool = False):
        ome_tiff_image = OMETIFFImage.load_from_path(path, switch_axis=switch_axis, is_annotation=is_annotation)
        warped_ome_tiff_image = ome_tiff_image.transform_data(registerer, transformation)
        return warped_ome_tiff_image
    
    @classmethod
    def load_data(cls, dct):
        path = dct['path']
        switch_axis = dct.get('switch_axis', False)
        is_annotation = dct.get('is_annotation', False)
        tif = tifffile.TiffFile(path)
        img = tif.asarray()
        if switch_axis and len(img.shape) > 2:
            img = np.moveaxis(img, 0, 2)
        if path.endswith('ome.tif') or path.endswith('ome.tiff'):
            is_ome = True
        else:
            is_ome = False            
        return cls(img, path, is_ome, tif, is_annotation, switch_axis)
        
    @classmethod
    def load_from_path(cls, path, keep_axis=False, is_annotation=False):
        tif = tifffile.TiffFile(path)
        img = tif.asarray()
        if is_annotation and not keep_axis and len(img.shape) > 2:
            img = np.moveaxis(img, 0, 2)
        if path.endswith('ome.tif') or path.endswith('ome.tiff'):
            is_ome = True
        else:
            is_ome = False
        print('image shape', img.shape)
        return cls(img, path, is_ome, tif, is_annotation, keep_axis)