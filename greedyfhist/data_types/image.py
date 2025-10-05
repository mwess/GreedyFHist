import abc
from dataclasses import dataclass, field
import os
from os import PathLike
from os.path import join
from typing import Iterable

from bioio import BioImage
from bioio_ome_tiff.writers import OmeTiffWriter
import numpy, numpy as np

from greedyfhist.registration.greedy_f_hist import GreedyFHist, RegistrationTransforms
from greedyfhist.utils.decorators import reset_scene_after_use
from greedyfhist.utils.io import read_bioio_image, write_to_ometiffile


INTERPOLATION_TYPE = int | str
INTERPOLATION_LIST_TYPE = list[INTERPOLATION_TYPE]
INTERPOLATION_DICT_TYPE = dict[str, INTERPOLATION_TYPE]
INTERPOLATION_LIST2_TYPE = list[INTERPOLATION_LIST_TYPE | INTERPOLATION_DICT_TYPE | None]
INTERPOLATION_DICT2_TYPE = dict[str, INTERPOLATION_LIST_TYPE | INTERPOLATION_DICT_TYPE | None]


def convert_list_to_dict(keys: list, values: list, default_fill: INTERPOLATION_TYPE = 0) -> dict:
    values = values + len(keys) * [default_fill]
    return {key: value for (key, value) in zip(keys, values)}


@dataclass
class InterpolationConfig:
    
    interpolate_default: INTERPOLATION_TYPE = 'LINEAR'
    interpolate_per_scene: INTERPOLATION_LIST_TYPE | INTERPOLATION_DICT_TYPE | None = None
    interpolate_per_channel: INTERPOLATION_LIST2_TYPE | INTERPOLATION_DICT2_TYPE | None = None

    
    @staticmethod
    def convert_interpolate_per_channel_to_dict(scene_channel_dict: dict,
                                                interpolate_per_channel: INTERPOLATION_LIST2_TYPE | INTERPOLATION_DICT2_TYPE | None,
                                                default_value: INTERPOLATION_TYPE) -> dict:
        if not interpolate_per_channel:
            return {}
        elif isinstance(interpolate_per_channel, list):
            new_dict = {}
            for scene in interpolate_per_channel:
                new_dict[scene] = {}
                channels_ = interpolate_per_channel[scene] # type: ignore
                if isinstance(channels_, dict):
                    new_dict[scene] = channels_
                else:
                    channel_dict = convert_list_to_dict(scene_channel_dict[scene], channels_, default_value)
                    new_dict[scene] = channel_dict
        else: # Assume that self.interpolate_per_channel is a dict
            new_dict = {}
            for scene in interpolate_per_channel:
                channels_ = interpolate_per_channel[scene]
                if isinstance(channels_, dict):
                    new_dict[scene] = channels_
                elif isinstance(channels_, list):
                    channel_dict = convert_list_to_dict(scene_channel_dict[scene], channels_, default_value)
                    new_dict[scene] = channel_dict
                elif isinstance(channels_, INTERPOLATION_TYPE):
                    channel_dict = convert_list_to_dict(scene_channel_dict[scene], [], default_value)
                    new_dict[scene] = channel_dict
                else:
                    raise ValueError('Input for interpolation cannot be parsed.')
        return new_dict
    
    def get_interpolation_modes(self, scenes: Iterable[str], channels: list[list[str]]) -> dict[str, dict[str, INTERPOLATION_TYPE]]:
        scenes = list(scenes)
        scene_channel_dict = {scene: channels_ for (scene, channels_) in zip(scenes, channels)}
        # Convert interpolate_per_scene and interpolate_per_channel to dicts
        if isinstance(self.interpolate_per_scene, list):
            self.interpolate_per_scene = convert_list_to_dict(scenes, self.interpolate_per_scene, self.interpolate_default)
        self.interpolate_per_channel = InterpolationConfig.convert_interpolate_per_channel_to_dict(scene_channel_dict,
                                                                                                 self.interpolate_per_channel,
                                                                                                 self.interpolate_default)
        interp_dict = {}
        for skey in scene_channel_dict:
            if skey not in interp_dict:
                interp_dict[skey] = {}
            for ckey in scene_channel_dict[skey]:
                if not self.interpolate_per_channel\
                   or (self.interpolate_per_channel\
                   and skey not in self.interpolate_per_channel\
                   and not ckey in self.interpolate_per_channel[skey]): # type: ignore
                    if self.interpolate_per_scene and skey in self.interpolate_per_scene:
                        interp_value = self.interpolate_per_scene[skey]
                    else:
                        interp_value = self.interpolate_default
                else:
                    interp_value = self.interpolate_per_channel[skey][ckey] # type: ignore
                interp_dict[skey][ckey] = interp_value
        return interp_dict
    
def default_interpolation_dict():
    return {'': {}}

@dataclass
class Image:
    
    img_data: BioImage
    path: str | PathLike
    # Use standard order of the BioImage if nothing else is added.
    order: str
    main_image: int | str = 0
    main_channel: int | str = 0
    reader: str | abc.ABCMeta | None = None
    is_ome: bool = False
    interpolation_modes: dict[str, dict[str, INTERPOLATION_TYPE]]  = field(default_factory=default_interpolation_dict)
    metadata: dict = field(default_factory=dict) 
    do_auto_squeeze: bool = False
    
    def __post_init__(self):
        try:
            self.img_data.ome_metadata
            is_ome = True
        except NotImplementedError:
            is_ome = False
        self.is_ome = is_ome
        if self.order is None:
            self.order = self.img_data.dims.order
        dim_order = self.order
        if self.is_ome:
            ome_xml = self.img_data.ome_metadata
        else:
            ome_xml = None
        channel_names = []
        image_names = []
        for _, img_name in enumerate(self.img_data.scenes):
            image_names.append(img_name)
            self.img_data.set_scene(img_name)
            channel_names_inner = []
            for _, channel_name in enumerate(self.img_data.channel_names):
                channel_names_inner.append(channel_name)
            channel_names.append(channel_names_inner)
        # Add function scaling to different physical sizes
        physical_pixel_sizes = self.img_data.physical_pixel_sizes
        metadata = {
            'physical_pixel_sizes': physical_pixel_sizes,
            'ome_xml': ome_xml,
            'image_names': image_names,
            'channel_names': channel_names,
            'dim_order': dim_order            
        }
        self.metadata = metadata
            
    @property
    @reset_scene_after_use
    def data(self):
        # Get channel idx
        if isinstance(self.main_channel , str):
            channel_names = self.img_data.get_channel_names()
            channel_idx = channel_names.index(self.main_channel)
        else:
            channel_idx = self.main_channel
        data = self.img_data.get_image_data(self.order, C=channel_idx)
        if self.do_auto_squeeze:
            data = data.squeeze()
        return data
        
    @reset_scene_after_use
    def transform_data(self, 
                       registerer: GreedyFHist, 
                       transformation: RegistrationTransforms) -> 'Image':
        current_scene = self.img_data.current_scene
        transformed_data_outer = []
        for img_idx, img_name in enumerate(self.img_data.scenes):
            self.img_data.set_scene(img_name)
            transformed_data_inner = []
            for channel_idx, channel_name in enumerate(self.img_data.channel_names):
                img_data = self.img_data.get_image_data(self.order, C=channel_idx)
                shape = img_data.shape
                redundant_dims = tuple(np.where(np.array(shape) == 1)[0])
                img_data = img_data.squeeze()
                interpolation = self.interpolation_modes[img_name][channel_name]
                transformed_data = registerer.transform_image(img_data, transformation, interpolation)
                transformed_data = np.expand_dims(transformed_data, redundant_dims)
                transformed_data_inner.append(transformed_data)
            axis_idx = self.order.index('C')
            transformed_data_inner = np.concatenate(transformed_data_inner, axis=axis_idx)
            transformed_data_outer.append(transformed_data_inner)

        # Set scene to state before transformation.
        self.img_data.set_scene(current_scene)

        transformed_bioio_image = BioImage(image=transformed_data_outer, 
                                           dim_order=self.order,
                                           channel_names=self.img_data.channel_names,
                                           physical_pixel_sizes=self.img_data.physical_pixel_sizes)
        new_image = Image(
            img_data=transformed_bioio_image,
            path=self.path,
            order=self.order,
            main_image=self.main_image,
            main_channel=self.main_channel,
            reader=self.reader,
            is_ome=self.is_ome,
            interpolation_modes=self.interpolation_modes,
            metadata=self.metadata,
            do_auto_squeeze=self.do_auto_squeeze
        )
        return new_image

    def to_directory(self, path: str | PathLike):
        fname = os.path.basename(self.path)
        name, _ = os.path.splitext(fname)
        if name.endswith('.ome'):
            name = name.rsplit('.ome', maxsplit=1)[0]
        path = join(path, f'{name}.ome.tif')
        self.to_file(path)

    def to_file(self, path: str | PathLike):
        OmeTiffWriter.save(
            data=self.data,
            uri=path,
            dim_order=self.order,
            ome_xml=self.metadata['ome_xml'] if self.is_ome else None,
            channel_names=self.metadata.get('channel_names', None),
            image_name=self.metadata.get('image_names', None),
            physical_pixel_sizes=self.metadata.get('physical_pixel_sizes', None)
        )

    @classmethod
    def load_from_path(cls, 
                       path: str | PathLike,
                       order: str | None = None,
                       main_image: int | str = 0,
                       main_channel: int | str = 0,
                       reader: str | abc.ABCMeta | None = None,
                       interpolation_config: InterpolationConfig | None = None,
                       do_auto_squeeze: bool = False):
            img_data = read_bioio_image(path, reader)
            scenes = img_data.scenes
            channels = []
            current_scene = img_data.current_scene
            for scene in scenes:
                img_data.set_scene(scene)
                channels.append(img_data.channel_names)
            img_data.set_scene(current_scene)
            if interpolation_config is None:
                interpolation_config = InterpolationConfig()
            interpolation_modes = interpolation_config.get_interpolation_modes(scenes, channels)
            if order is None:
                order = img_data.dims.order
            return cls(img_data=img_data,
                       path=path,
                       order=order, # type: ignore
                       main_image=main_image,
                       main_channel=main_channel,
                       reader=reader,
                       interpolation_modes=interpolation_modes,
                       do_auto_squeeze=do_auto_squeeze
                       )
    
    @classmethod
    def load_from_config(cls, config: dict):
        path = config['path']
        order = config.get('order', None)
        main_image = config.get('main_image', 0)
        main_channel = config.get('main_channel', 0)
        reader = config.get('reader', '')
        interpolation_config = config.get('interpolation_config', None)
        do_auto_squeeze = config.get('do_auto_squeeze', False)
        return cls.load_from_path(
            path=path,
            order=order,
            main_image=main_image,
            main_channel=main_channel,
            reader=reader,
            interpolation_config=interpolation_config,
            do_auto_squeeze=do_auto_squeeze
        )
        
    @staticmethod
    def write_numpy_to_ometiff(image: numpy.ndarray,
                               path: str | PathLike):
        bimg = BioImage(image)
        