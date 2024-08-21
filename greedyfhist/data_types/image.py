from dataclasses import dataclass
import os
from typing import Dict, Optional, Union
import xml.etree.ElementTree as ET

import numpy
import numpy as np
import pyvips
import SimpleITK as sitk
import tifffile

from greedyfhist.registration.greedy_f_hist import GreedyFHist, RegistrationTransforms


def write_to_ometiffile(img: numpy.array, 
                     path: str, 
                     metadata: Dict = None, 
                     is_annotation: bool = False,
                     tile: bool = True,
                     tile_size: int = 512,
                     pyramid: bool = True):
    if metadata is None:
        metadata = {
            'PhysicalSizeX': 1,
            'PhysicalSizeXUnit': 'px',
            'PhysicalSizeY': 1,
            'PhysicalSizeYUnit': 'px',
            'Interleaved': 'false'
        }
    # Taken from https://forum.image.sc/t/writing-qupath-bio-formats-compatible-pyramidal-image-with-libvips/51223/6
    if len(img.shape) == 3:
        w, h, c = img.shape
    else:
        c = 1
        w, h = img.shape
    if is_annotation:
        c, w, h = w, h, c
    img_vips = pyvips.Image.new_from_array(img)
    xml_string = f"""<?xml version="1.0" encoding="UTF-8"?>
    <OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"
        xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
        xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd">
        <Image ID="Image:0">
            <!-- Minimum required fields about image dimensions -->
            <Pixels DimensionOrder="XYCZT"
                    ID="Pixels:0"
                    PhysicalSizeX="{metadata['PhysicalSizeX']}"
                    PhysicalSizeXUnit="{metadata['PhysicalSizeXUnit']}"
                    PhysicalSizeY="{metadata['PhysicalSizeY']}"
                    PhysicalSizeYUnit="{metadata['PhysicalSizeYUnit']}"
                    Interleaved="{metadata.get('Interleaved', True)}"
                    SizeC="{c}"
                    SizeT="1"
                    SizeX="{h}"
                    SizeY="{w}"
                    SizeZ="1"
                    Type="uint8">
            </Pixels>
        </Image>
        <StructuredAnnotations>
		<MapAnnotation ID="Annotation:Resolution:0" Namespace="openmicroscopy.org/PyramidResolution"><Value/></MapAnnotation></StructuredAnnotations>
    </OME>"""
    root = ET.fromstring(xml_string)
    ns = {'ns0': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
    elem = root.find('*/ns0:Pixels', ns)
    for channel_ in metadata['channels']:
        elem.append(channel_)
    for tiff_data_ in metadata['tiff_data']:
        elem.append(tiff_data_)
    ome_metadata_xml_string = ET.tostring(root, encoding='unicode')
    img_vips.set_type(pyvips.GValue.gint_type, "page-height", img_vips.height)
    img_vips.set_type(pyvips.GValue.gstr_type, "image-description", ome_metadata_xml_string)
    img_vips.tiffsave(path, tile=tile, tile_width=tile_size, tile_height=tile_size, pyramid=pyramid)
    return ome_metadata_xml_string


def get_metadata_from_tif(xml_string: str) -> Dict:
    root = ET.fromstring(xml_string)
    ns = {'ns0': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
    elem = root.findall('*/ns0:Pixels', ns)[0]
    metadata = elem.attrib
    channels = root.findall('**/ns0:Channel', ns)
    tiff_data = root.findall('**/ns0:TiffData', ns)
    metadata['channels'] = channels
    metadata['tiff_data'] = tiff_data
    return metadata


def is_tiff_file(suffix: str) -> bool:
    return suffix in ['.tif', '.tiff']


def read_image(path: str, is_annotation: bool = False) -> Union[numpy.array, Optional[Dict]]:
    suffix = os.path.splitext(path)[1]
    metadata = None
    if is_tiff_file(suffix):
        tif = tifffile.TiffFile(path)
        img = tif.asarray()
        if tif.ome_metadata is not None:
            metadata = get_metadata_from_tif(tif.ome_metadata)
    elif suffix in pyvips.base.get_suffixes():
        img_vips = pyvips.Image.new_from_file(path)
        img = img_vips.numpy()
        image_description = img_vips.get('image-description')
        if is_tiff_file(suffix):
            metadata = get_metadata_from_tif(image_description)
    elif path.endswith('.nii.gz') or path.endswith('.nii'):
        img = sitk.GetArrayFromImage(sitk.ReadImage(path))
    return img, metadata


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

    @staticmethod
    def load_and_transform_data(path: str, 
                                registerer: GreedyFHist,
                                transformation: RegistrationTransforms,
                                keep_axis: bool = True,
                                is_annotation: bool = False):
        image = Image.load_from_path(path, keep_axis=keep_axis, is_annotation=is_annotation)
        warped_image = image.transform_data(registerer, transformation)
        return warped_image

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