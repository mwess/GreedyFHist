from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class Options:

    resolution: Tuple[int, int] = field(default=(1024, 1024))
    kernel_size: int = 10
    output_directory: str = 'out'
    temporary_directory: str = 'tmp'
    affine_do_registration: bool = True
    deformable_do_registration: bool = True
    affine_do_denoising: bool = True
    deformable_do_denoising: bool = True
    moving_segmentation_mask = None
    fixed_segmentation_mask = None
    moving_sr: int = 30
    moving_sp: int = 20
    fixed_sr: int = 30
    fixed_sp: int = 20
    # padding: int
    pre_downsampling_factor: float = 1
    store_cmdline_returns: bool = True
    remove_temporary_directory: bool = True
    greedy_opts: 'GreedyOptions' 

    
    def __post_init__(self):
        self.greedy_opts = GreedyOptions()
    
    def parse_dict(self, args_dict):
        # First collect all greedy parameters
        greedy_args = self.__collect_greedy_args(args_dict)
        for key in args_dict:
            self.__assign_if_present(key, args_dict)
        self.greedy_opts.parse_dict(greedy_args)

    def __collect_greedy_args(self, args_dict):
        greedy_args = {}
        for key in args_dict:
            if key.startswith('greedy_'):
                g_key = key.lstrip('greedy_')
                value = args_dict[key]
                greedy_args[g_key] = value
        return greedy_args
        

    def __assign_if_present(self, key, args_dict):
        if key in args_dict:
            value = args_dict[key]
            if key in self.__anotations__:
                self.__setattribute__(key, value)

    def to_dict(self):
        d = {}
        for key in self.__annotations__:
            value = self.__getattribute__[key]
            if isinstance(value, GreedyOptions):
                d[key] = value.to_dict()
            else:
                d[key] = value
        return d


@dataclass
class GreedyOptions:

    dim: int = 2
    s1: float = 6.0
    s2: float = 5.0
    kernel_size: int = 10
    cost_function: str = 'ncc'
    iteration_rigid: int = 10000
    ia: str = 'ia_com_init'
    pyramid_iterations: List[int] = field(default=[100, 50, 10])
    n_threads: int = 1
    use_sv: bool = False
    use_svlb: bool = False

    def parse_dict(self, args_dict):
        for key in args_dict:
            self.__assign_if_present(key, args_dict)

    def __assign_if_present(self, key, args_dict):
        if key in args_dict:
            value = args_dict[key]
            if key in self.__anotations__:
                self.__setattribute__(key, value)

    def to_dict(self):
        d = {}
        for key in self.__annotations__:
            d[key] = self.__getattribute__[key]
        return d
            
