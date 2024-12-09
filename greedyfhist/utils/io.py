"""
Contains io helper functions
"""
import os
from os.path import join
from pathlib import Path
import shutil
from typing import Optional
import xml.etree.ElementTree as ET


import numpy, numpy as np
import pyvips
import SimpleITK as sitk
import tifffile


def create_if_not_exists(path: str) -> None:
    """Creates directory, if it does not exist. Clashes with existing directories result in nothing getting created.

    Args:
        path (str):
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def clean_if_exists(path: str):
    """Remove directory, if it exists.

    Args:
        path (str): 
    """
    if os.path.exists(path):
        shutil.rmtree(path)


def write_mat_to_file(mat: numpy.ndarray, fname: str) -> None:
    """Writes 2d affine matrix to file. File format is human readable.

    Args:
        mat (numpy.ndarray): 
        fname (str):
    """
    out_str = f'{mat[0,0]} {mat[0,1]} {mat[0,2]}\n{mat[1,0]} {mat[1,1]} {mat[1,2]}\n{mat[2,0]} {mat[2,1]} {mat[2,2]}'
    with open(fname, 'w') as f:
        f.write(out_str)


def derive_output_path(directory: str, fname: str, limit: int = 1000) -> str:
    """Generates a unique output path. If path is already existing,
    adds a counter value until a unique path is found.

    Args:
        directory (str): target directory
        fname (str): target filename
        limit (int, optional): Limit number to prevent endless loops. Defaults to 1000.

    Returns:
        str: Target path
    """
    target_path = join(directory, fname)
    if not os.path.exists(target_path):
        return target_path
    for suffix in range(limit):
        new_target_path = f'{target_path}_{suffix}'
        if not os.path.exists(new_target_path):
            return new_target_path
    return target_path
        
def get_default_metadata():
    metadata = {
        'PhysicalSizeX': 1,
        'PhysicalSizeXUnit': 'px',
        'PhysicalSizeY': 1,
        'PhysicalSizeYUnit': 'px',
        'Interleaved': 'true',
        'channels': [],
        'tiff_data': []
    }
    return metadata


def write_to_ometiffile(img: numpy.ndarray, 
                     path: str, 
                     metadata: dict = None, 
                     is_annotation: bool = False,
                     tile: bool = True,
                     tile_size: int = 512,
                     pyramid: bool = True,
                     bigtiff: bool = False,
                     skip_channel: bool = False,
                     skip_tiffdata: bool = False):
    """
    Writes an image as an ometif file. If the image to store was 
    originally extracted as an ome.tif, the metadata from the 
    source ome.tif can be passed as well. For multichannel annotations,
    the format C x W x H is expected, otherwise W x H x C, or W x H.  

    Args:
        img (numpy.ndarray): Image to be stored. 
        path (str): Output path.
        metadata (Dict): Contains metadata for ome.tif. Only used if
                         images original source was also tif. If None,
                         a simple default dict is used.
        is_annotation (bool): If True, assumes that image is C x W x H.
                              Otherwise assumes, W x H x C. 
        tile (bool): Use tiling. Defaults to True.
        tile_size: Size of tile. Defaults to 512.
        pyramid (bool): Build pyramidical. Defaults to True.
    """
    if not metadata:
        metadata = {
            'PhysicalSizeX': 1,
            'PhysicalSizeXUnit': 'px',
            'PhysicalSizeY': 1,
            'PhysicalSizeYUnit': 'px',
            'Interleaved': 'true',
            'channels': [],
            'tiff_data': []
        }
    if is_annotation:
        metadata['Interleaved'] = 'false'
    ns = {'ns0': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
    if (path.endswith('.ome.tif') or path.endswith('.ome.tiff')) and not skip_tiffdata and 'tiff_data' in metadata:
        target_fname = os.path.basename(path)
        for tiff_data in metadata['tiff_data']:
            for uuid_node in tiff_data.findall('ns0:UUID', ns):
                uuid_node.attrib['FileName'] = target_fname
    # Taken from https://forum.image.sc/t/writing-qupath-bio-formats-compatible-pyramidal-image-with-libvips/51223/6
    if len(img.shape) == 3:
        w, h, c = img.shape
    else:
        c = 1
        w, h = img.shape
    if is_annotation and len(img.shape) == 3:
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
    if not skip_channel:
        channels = metadata.get('channels', [])
        for channel_ in channels:
            elem.append(channel_)
    if not skip_tiffdata:
        tiff_data = metadata.get('tiff_data', [])
        for tiff_data_ in tiff_data:
            elem.append(tiff_data_)
    ome_metadata_xml_string = ET.tostring(root, encoding='unicode')
    img_vips.set_type(pyvips.GValue.gint_type, "page-height", img_vips.height)
    img_vips.set_type(pyvips.GValue.gstr_type, "image-description", ome_metadata_xml_string)
    img_vips.tiffsave(path, tile=tile, tile_width=tile_size, tile_height=tile_size, pyramid=pyramid, bigtiff=bigtiff)


def is_tiff_file(suffix: str) -> bool:
    return suffix in ['.tif', '.tiff']


def read_image(path: str, is_annotation: bool = False) -> numpy.ndarray | Optional[dict]:
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
    if is_annotation and len(img.shape) == 3:
        img = np.moveaxis(img, 0, 2)
    return img, metadata


def get_metadata_from_tif(xml_string: str) -> dict:
    """
    Extracts metadata from tiffile.

    Args:
        xml_string (str): Tif metadata xml as a string.
    
    Returns:
        Dictionary with extracted metadata.
    """
    root = ET.fromstring(xml_string)
    ns = {'ns0': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
    elem = root.findall('*/ns0:Pixels', ns)[0]
    metadata = elem.attrib
    channels = root.findall('**/ns0:Channel', ns)
    tiff_data = root.findall('**/ns0:TiffData', ns)
    metadata['channels'] = channels
    metadata['tiff_data'] = tiff_data
    if 'PhysicalSizeX' in metadata:
        metadata['PhysicalSizeX'] = float(metadata['PhysicalSizeX'])
    if 'PhysicalSizeY' in metadata:
        metadata['PhysicalSizeY'] = float(metadata['PhysicalSizeY'])
    del metadata['SizeX']
    del metadata['SizeY']
    del metadata['SizeC']
    return metadata