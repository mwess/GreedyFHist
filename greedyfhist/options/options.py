from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union


def get_3_step_pyramid_iterations():
    return [100, 50, 10]

def get_4_step_pyramid_iterations():
    return [100, 100, 50, 10]

@dataclass
class GreedyOptions:

    dim: int = 2
    s1: float = 5.0
    s2: float = 5.0
    kernel_size: int = 10
    cost_function: str = 'ncc'
    iteration_rigid: int = 10000
    ia: str = 'ia-com-init'
    affine_iteration_pyramid: List[int] = field(default_factory=get_3_step_pyramid_iterations)
    nonrigid_iteration_pyramid: List[int] = field(default_factory=get_4_step_pyramid_iterations)
    n_threads: int = 1
    use_sv: bool = False
    use_svlb: bool = False
    exp: Optional[int] = None
    # TODO: Parse this option correctly.
    yolo_segmentation_min_size=5000

    def parse_dict(self, args_dict):
        for key in args_dict:
            self.__assign_if_present(key, args_dict)

    def __assign_if_present(self, key, args_dict):
        if key in args_dict:
            value = args_dict[key]
            if key in self.__annotations__:
                self.__setattr__(key, value)

    def to_dict(self):
        d = {}
        for key in self.__annotations__:
            d[key] = self.__getattribute__(key)
        return d
            
    @staticmethod
    def default_options():
        return GreedyOptions()

def load_greedyoptions():
    return GreedyOptions()

def load_default_resolution():
    return (1024, 1024)

@dataclass
class RegistrationOptions:

    greedy_opts: 'GreedyOptions' = field(default_factory=GreedyOptions.default_options)
    resolution: Tuple[int, int] = field(default_factory=load_default_resolution)
    do_affine_registration: bool = True
    do_nonrigid_registration: bool = True
    enable_affine_denoising: bool = True
    enable_deformable_denoising: bool = True
    moving_sr: int = 30
    moving_sp: int = 25
    fixed_sr: int = 30
    fixed_sp: int = 25
    pre_downsampling_factor: Union[float, str] = 1
    keep_affine_transform_unbounded: bool = True
    temporary_directory: str = 'tmp'
    remove_temporary_directory: bool = True
    
    def __post_init__(self):
        self.greedy_opts = GreedyOptions()
    
    def __assign_if_present(self, key, args_dict):
        if key in args_dict:
            value = args_dict[key]
            if key in self.__annotations__:
                self.__setattr__(key, value)
    
    def to_dict(self):
        d = {}
        for key in self.__annotations__:
            value = self.__getattribute__(key)
            if isinstance(value, GreedyOptions):
                d[key] = value.to_dict()
            else:
                d[key] = value
        return d

    @staticmethod
    def parse_cmdln_dict(args_dict):
        opts = RegistrationOptions()
        for key in args_dict:
            if key in ['greedy', 'resolution']:
                continue
            opts.__assign_if_present(key, args_dict)
        if 'resolution' in args_dict:
            resolution_str = args_dict['resolution']
            resolution = resolution_str.split('x')
            resolution = (int(resolution[0]), int(resolution[1]))
            opts.resolution = resolution
        if 'greedy' in args_dict:
            greedy_args = args_dict['greedy']
            opts.greedy_opts.parse_dict(greedy_args)
        return opts

    @staticmethod
    def default_options():
        return RegistrationOptions()