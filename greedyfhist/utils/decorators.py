from collections import OrderedDict
import functools

from bioio import BioImage

def reset_scene_after_use(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        var_name = 'img_data'
        if not var_name in self.__dict__ and not isinstance(self.__getattribute__(var_name), BioImage):
            raise AttributeError('Decorator cannot be applied on this object.')
        img_data: BioImage = self.__getattribute__(var_name)
        current_scene = img_data.current_scene
        res = func(self, *args, **kwargs)
        img_data.set_scene(current_scene)
        return res
    return wrapper
