"""This module contains all the options classes for configuring the registration.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy

from .segmentation_options import (
    SegmentationOptions,
    YoloSegOptions,
    TissueEntropySegOptions,
    LuminosityAndAreaSegOptions,
    parse_segmentation_options
)


def _get_3_step_pyramid_iterations() -> list[int]:
    """Returns a default 3-step pyramid with iterations 100, 50 and 10.

    Returns:
        list[int]: pyramid with iterations
    """
    return [100, 50, 10]


def _get_4_step_pyramid_iterations() -> list[int]:
    """Returns a default 4-step pyramid with iterations 100, 100, 50 and 10.

    Returns:
        list[int]:
    """
    return [100, 100, 50, 10]


def _get_5_step_pyramid_iterations() -> list[int]:
    """Returns a default 5-step pyramid with iterations 100, 100, 50, 50 and 10.

    Returns:
        list[int]:
    """
    return [100, 100, 50, 50, 10]


def _load_default_resolution() -> tuple[int, int]:
    """Loads the default resolution of 1024 x 1024.

    Returns:
        tuple[int, int]:
    """
    return 1024, 1024

    
def _load_default_nr_resolution() -> tuple[int, int]:
    """Loads the default resolution of 2048 x 2048 
    for nonrigid registration.

    Returns:
        tuple[int, int]: 
    """
    return 2048, 2048


@dataclass
class PreprocessingOptions:
    """
    Contains all the options that can be used to register a moving to a fixed image.
        
    moving_sr: int = 30
        Color window radius for mean shift filtering in moving image.

    moving_sp: int = 25
        Pixel window radius for mean shift filtering in moving image.
        
    fixed_sr: int = 30
        Color window radius for mean shift filtering in fixed image.
        
    fixed_sp: int = 25
        Pixel window radius for mean shift filtering in fixed image.

    enable_denoising: bool = True
        Toggle for enabling denoising.
        
    disable_denoising_moving: bool = False
        Can be used to disable denoising in moving image.
        
    disable_denoising_fixed: bool = False
        Can be used to disable denoising in fixed image.       
    """

    moving_sr: int = 30
    moving_sp: int = 25
    fixed_sr: int = 30
    fixed_sp: int = 25
    enable_denoising: bool = True
    disable_denoising_moving: bool = False
    disable_denoising_fixed: bool = False
    
    
    def __assign_if_present(self, key, args_dict):
        """Assigns value of given key in args_dict if key in class's __annotations__.

        Args:
            key (_type_):
            args_dict (_type_):
        """
        if key in args_dict:
            value = args_dict[key]
            if key in self.__annotations__:
                self.__setattr__(key, value)

    def to_dict(self):
        d = {}
        for key in self.__annotations__:
            d[key] = self.__getattribute__(key)
        return d
    
    def parse_dict(self, args_dict: dict):
        """Function made to automatically parse attributes from dictionary. 

        Args:
            args_dict (Dict): 
        """
        for key in args_dict:
            self.__assign_if_present(key, args_dict)    

    @staticmethod
    def default_options() -> PreprocessingOptions:
        return PreprocessingOptions()

    @staticmethod
    def default_options_nr() -> PreprocessingOptions:
        """Default options for nonrigid registration.

        Returns:
            PreprocessingOptions: 
        """
        options = PreprocessingOptions()
        options.enable_denoising = False
        return options
    

@dataclass
class AffineGreedyOptions:
    """
    Commandline options parsed to Greedy. 

    More detailed information can be found in Greedy's documentation.
    Some options are extended/overwritten.

    Attributes:
    
        dim: int 
            Should always be set to 2.

        resolution: Tuple[int, int]
            Image resolution after downscaling for Greedy.  

        preprocessing_options: PreprocessingOptions
            Options for preprocessing.          
            
        kernel_size: int
            kernel size for 'ncc' kernel metric.
            
        cost_function: str
            cost function used to optimize the registration. Should always be set to 'ncc'.
            
        rigid_iterations: Union[int, str]
            Number of rigid iterations during initial registration of affine registration.
            If set to 'auto', the number of iterations for rigid matching
            is computed relative to the size of the offset from the 
            center-of-mass initialization. Otherwise, uses the provided
            number. 
            
        ia: str
            Initial image alignment. 'ia-com-init' is a custom option for using the center-of-mass of image masks. Other options can be taken from Greedy. 
            
        iteration_pyramid: List[int]
            Iterations in the multiresolution pyramid.
            
        n_threads: int
            Number of threads. Defaults to 1, but 8 has shown the fastest registrations.
            
        keep_affine_transform_unbounded: bool = True
            If true, keeps affine transform unbounded. Otherwise, affine
            transform is translated into a displacement field. Should be
            set to True.            

        dof: int = 12
            Degrees of freedom for registration. (6 = rigid, 12 = affine)
    """

    dim: int = 2
    resolution: tuple[int, int] = field(default_factory=_load_default_resolution)
    preprocessing_options: 'PreprocessingOptions' = field(default_factory=PreprocessingOptions.default_options)
    kernel_size: int = 10
    cost_function: str = 'ncc'
    rigid_iterations: int | str = 10000
    ia: str = 'ia-com-init'
    iteration_pyramid: list[int] = field(default_factory=_get_4_step_pyramid_iterations)
    n_threads: int = 1
    keep_affine_transform_unbounded: bool = True
    dof: int = 12

    def __post_init__(self):
        self.preprocessing_options = PreprocessingOptions()

    def parse_dict(self, args_dict: dict):
        """Function made to automatically parse attributes from dictionary. 

        Args:
            args_dict (Dict): 
        """
        for key in args_dict:
            if key in ['preprocessing_options']:
                continue
            self.__assign_if_present(key, args_dict)
        if 'preprocessing_options' in args_dict:
            self.preprocessing_options.parse_dict(args_dict['preprocessing_options'])

    def __assign_if_present(self, key: Any, args_dict: dict):
        """Assigns value if key is present in class's __annotations__.

        Args:
            key (Any):
            args_dict (Dict):
        """
        if key in args_dict:
            value = args_dict[key]
            if key in self.__annotations__:
                self.__setattr__(key, value)

    def to_dict(self) -> dict:
        """Return options as dictionary.

        Returns:
            Dict:
        """
        d = {}
        for key in self.__annotations__:
            value = self.__getattribute__(key)
            if isinstance(value, PreprocessingOptions):
                d[key] = value.to_dict()
            else:
                d[key] = value
        return d
            
    @staticmethod
    def default_options() -> 'AffineGreedyOptions':
        """Returns default options.

        Returns:
            AffineGreedyOptions:
        """
        return AffineGreedyOptions()


@dataclass
class NonrigidGreedyOptions:
    """
    Commandline options parsed to Greedy. 

    More detailed information can be found in Greedy's documentation.
    Some options are extended/overwritten.

    Attributes:
    
        dim: int 
            Should always be set to 2.

        resolution: Tuple[int, int]
            Image resolution after downscaling for Greedy.  

        preprocessing_options: PreprocessingOptions
            Options for preprocessing.          
            
        s1: float
            pre sigma value for nonrigid registration.
            
        s2: float
            post sigma value for nonrigid registration.
            
        kernel_size: int
            kernel size for 'ncc' kernel metric.
            
        cost_function: str
            cost function used to optimize the registration. Should always be set to 'ncc'.
            
        ia: str
            Initial image alignment. 'ia-com-init' is a custom option for using the center-of-mass of image masks. Other options can be taken from Greedy. 
            
        iteration_pyramid: List[int] 
            Iterations in the multiresolution pyramid.
            
        n_threads: int
            Number of threads. Defaults to 1, but 8 has shown the fastest registrations.
            
        use_sv: bool
            Additional greedy option for nonrigid registration.
            
        use_svlb: bool
            Additional experimental greedy option for nonrigid registration.
            
        exp: Optional[int]
            Additional value used in conjunction with use_svlb.
            
        use_gm_trim: bool = True
            Passes the gm_trim param to greedy.
            
        tscale: str | None = None
            If not None, passes tscale to greedy.
    """

    dim: int = 2
    resolution: tuple[int, int] = field(default_factory=_load_default_nr_resolution)
    preprocessing_options: PreprocessingOptions = field(default_factory=PreprocessingOptions.default_options_nr)    
    s1: float = 5
    s2: float = 4
    kernel_size: int = 10
    cost_function: str = 'ncc'
    ia: str = 'ia-com-init'
    iteration_pyramid: list[int] = field(default_factory=_get_4_step_pyramid_iterations)
    n_threads: int = 1
    use_sv: bool = False
    use_svlb: bool = False
    exp: int | None = None
    use_gm_trim: bool = True
    tscale: str | None = None

    def __post_init__(self):
        self.preprocessing_options.enable_denoising = False

    def parse_dict(self, args_dict: dict):
        """Function made to automatically parse attributes from dictionary. 

        Args:
            args_dict (Dict): 
        """
        for key in args_dict:
            if key in ['preprocessing_options']:
                continue
            self.__assign_if_present(key, args_dict)
        if 'preprocessing_options' in args_dict:
            self.preprocessing_options.parse_dict(args_dict['preprocessing_options'])

    def __assign_if_present(self, key: Any, args_dict: dict):
        """Assigns value if key is present in class's __annotations__.

        Args:
            key (Any):
            args_dict (Dict):
        """
        if key in args_dict:
            value = args_dict[key]
            if key in self.__annotations__:
                self.__setattr__(key, value)
        
    def to_dict(self) -> dict:
        """Return options as dictionary.

        Returns:
            Dict:
        """
        d = {}
        for key in self.__annotations__:
            value = self.__getattribute__(key)
            if isinstance(value, PreprocessingOptions):
                d[key] = value.to_dict()
            else:
                d[key] = value
        return d
            
    @staticmethod
    def default_options() -> 'NonrigidGreedyOptions':
        """Returns default options.

        Returns:
            GreedyOptions:
        """
        return NonrigidGreedyOptions()
    
    @staticmethod
    def default_nrpt_options() -> 'NonrigidGreedyOptions':
        """Returns suitable default options for nrpt registration.

        Returns:
            NonrigidGreedyOptions:
        """
        opts = NonrigidGreedyOptions()
        opts.resolution = (1024, 1024)
        return opts


@dataclass
class TilingOptions:
    """Options for performing nonrigid tiling registration.
    
    Attributes:
    
        enable_tiling: bool = False
            Enables tiling. If False, standard non-rigid registration is used.
            
        tiling_mode: str = 'simple'
            Tiling mode. Can either be 'simple' or 'pyramid'. 
            
        stop_condition_tile_resolution: bool = False
            Relevant for pyramid tiling. One condition for stopping pyramid. If the size of 
            a tile (without overlapping) if smaller than the downscaling resolution during 
            registration, the pyramid is stopped. Ignored if `tiling_mode=='simple'`.
            
        stop_condition_pyramid_counter: bool = True
            Relevant for pyramid tiling. One condition for stopping pyramid. Stops as soon 
            as the pyramid depth reaches `max_pyramid_depth`. Ignored if `tiling_mode=='simple'`.
            
        max_pyramid_depth: int | None = 0
            Relevant for pyramid tiling. Defines stop condition for `stop_condition_pyramid_counter`.
            
        pyramid_resolutions: list[int] | None = None
            Relevant for pyramid tiling. Defines the resolution of the tiles at each step of the pyramid.
        
        pyramid_tiles_per_axis: list[int] | None = None
            Relevant for pyramid tiling. Defines how many tiles are generated per axis. The number of tiles
            is then always the number of tiles on the x-axis * number of tiles on the y-axis.
        
        tile_overlap: list[float] | float = 0.75
            Relevant for pyramid tiling. Gives the overlap of two neighboring tiles.
            
        tile_size: int | tuple[int, int] = 1024
            Relevant for simple tiling. Size of each tile extracted. 
            
        min_overlap: float = 0.1
            Relevant for simple tiling. Minimum overlap between two neighboring tiles. Might be larger depending of how 
            tiles fit into the images resolution. Last tile is likely to be affected if the tiles can not be evenly 
            extracted from the image.
            
        n_procs: int | None = None
            Number of concurrent processes for tile registration. If None, tiles are registered sequentially.
    """
    
    enable_tiling: bool = False
    tiling_mode: str = 'simple'    
    stop_condition_tile_resolution: bool = False
    stop_condition_pyramid_counter: bool = True
    max_pyramid_depth: int | None = 0
    pyramid_resolutions: list[int] | None = None
    pyramid_tiles_per_axis: list[int] | None = None
    tile_overlap: list[float] | float = 0.75
    tile_size: int | tuple[int, int] = 1024
    min_overlap: float = 0.1
    n_procs: int | None = None
    
    @staticmethod
    def default_options() -> 'TilingOptions':
        """Returns default options.

        Returns:
            TilingOptions: 
        """
        return TilingOptions()

    def __assign_if_present(self, key, args_dict):
        """Assigns value of given key in args_dict if key in class's __annotations__.

        Args:
            key (_type_):
            args_dict (_type_):
        """
        if key in args_dict:
            value = args_dict[key]
            if key in self.__annotations__:
                self.__setattr__(key, value)
    
    def parse_dict(self, args_dict: dict):
        """Function made to automatically parse attributes from dictionary. 

        Args:
            args_dict (Dict): 
        """
        for key in args_dict:
            self.__assign_if_present(key, args_dict)


@dataclass
class RegistrationOptions:
    """
    Contains all the options that can be used to register a moving to a fixed image.
    
    path_to_greedy: str | None = None
        Path to greedy executable. Only needed in the functional interface.
        In the object-oriented interface this property is set to 
        `GreedyFHist.path_to_greedy` if necessary.
        
    segmentation: SegmentationOptions | Callable[[numpy.ndarray], numpy.ndarray] | str | None
        Handles segmentation accordingly:
        SegmentationOptions:
            Contains configuration for YOLO8 based segmentations.
        Callable:
            Excepts a segmentation function that takes an input image and returns a mask where
            1 denotes tissue area and 0 denotes background.
        str:
            One of 'yolo-seg', 'entropy-based-seg', 'lum-area-seg'.
            If 'yolo-seg', the function `load_yolo_segmentation` is called to init a segmentation
            function based on the yolo model.
            
            If 'entropy-based-seg', the function `load_tissue_entropy_detection` is called with default
            parameters and `predict_tissue_from_entropy` function is loaded.
            
            If 'lum-area-seg', the function `load_tissue_luminosity_area_detection` is called with 
            default values and `predict_tissue_from_luminosity_and_area` function is loaded.
        None:
            Loads yolo based segmentation with default options.
            
        Either an options object containing configuration parameters for the standard
        tissue segmentation or a segmentation function that takes an image an input
        and returns a mask.
        
    use_docker_container: bool = False
        If the greedy executable is called through a docker image, set this to True and 
        set `path_to_greedy` to the name of the docker container with the greedy executable.
        
    affine_registration_options: AffineRegistrationOptions
        Options for preprocessing and calling Greedy with affine
        registration otions.

    nonrigid_registration_options: NonrigidRegistrationOptions
        Options for preprocessing and calling Greedy with nonrigid
        registration options.

    pre_sampling_factor: Union[float, str] = 'auto'
        Sampling factor prior to preprocessing. Does not
        affect registration accuracy, but can help to speed up the  
        preprocessing considerable, especially for large images. If
        the factor is a float it is interpreted as a scaling factor
        that is applied on both images prior to preprocessing. If
        'auto', then the `pre_sampling_max_img_size` is used to
        scale both images to have a maximum resolution of that factor.
        
    pre_sampling_max_img_size: Optional[int] = 2000
        Resizes moving and fixed image such that the largest axis
        of both images is set to a maximum of `pre_sampling_resize_factor`.
        This option is used if `pre_sampling_factor` is set to auto. 

    do_affine_registration: bool = True
        Whether an affine registration is performed or not.
        
    do_nonrigid_registration: bool = True
        Whether a deformable registration is performed or not.     

    do_nrpt_registration: bool = False
        If True, will perform nonrigid-pyramidic tiling registration. Will 
        automatically set `do_affine_registration` to False and use 
        `nonrigid_registration_options` for registration of tiles.

    compute_reverse_nonrigid_registration: bool = False
        Compute the reverse nonrigid registration. If do_affine_registration
        is True, uses the inverse of affine transformation as an
        initialization, if affine registration is used. This options is typically
        used for groupwise registration if the reverse transform is needed as well.    
        
    keep_affine_transform_unbounded: bool = True
        If true, keeps affine transform unbounded. Otherwise, affine
        transform is translated into a displacement field. Should be
        set to True.
        
    temporary_directory: str = 'tmp'
        Temporary directory used for storing Greedy in- and output.
        
    remove_temporary_directory: bool = True
        Sets whether the temporary directory is removed after
        registration.    
        
    disable_mask_generation: bool = False
        If True, does not generation masks. Internally, the whole image area is declared
        as a mask. Does nothing if masks are provided.
        
    tiling_options: TilingOptions
        Options for defining tiling options in nrpt registration.
        
    grp_n_proc: int | None = None
        Number of concurrent processes used during groupwise registration. If None, performs
        every registration sequentially.
    """

    path_to_greedy: str | None = None
    segmentation: SegmentationOptions | Callable[[numpy.ndarray], numpy.ndarray] | str = field(default_factory=YoloSegOptions.default_options)
    use_docker_container: bool = False
    affine_registration_options: AffineGreedyOptions = field(default_factory=AffineGreedyOptions.default_options)
    nonrigid_registration_options: NonrigidGreedyOptions = field(default_factory=NonrigidGreedyOptions.default_options)
    pre_sampling_factor: float | str = 'auto'
    pre_sampling_max_img_size: int | None = 2000    
    do_affine_registration: bool = True
    do_nonrigid_registration: bool = True
    compute_reverse_nonrigid_registration: bool = False
    temporary_directory: str = 'tmp'
    remove_temporary_directory: bool = True
    disable_mask_generation: bool = False
    tiling_options: TilingOptions = field(default_factory=TilingOptions.default_options)
    grp_n_proc: int | None = None
    
    def __assign_if_present(self, key, args_dict):
        """Assigns value of given key in args_dict if key in class's __annotations__.

        Args:
            key (_type_):
            args_dict (_type_):
        """
        if key in args_dict:
            value = args_dict[key]
            if key in self.__annotations__:
                self.__setattr__(key, value)
    
    def to_dict(self):
        d = {}
        for key in self.__annotations__:
            value = self.__getattribute__(key)
            if isinstance(value, AffineGreedyOptions):
                d[key] = value.to_dict()
            elif isinstance(value, NonrigidGreedyOptions):
                d[key] = value.to_dict()
            else:
                d[key] = value
        return d

    @staticmethod
    def parse_cmdln_dict(args_dict: dict) -> 'RegistrationOptions':
        """Sets all values in dictionary that have a matching key in 
        class's __annotation__ field. key/value pairs for GreedyOpts
        are put in a sub dict, called 'greedy'. 'resolution' is in
        string, e.g. '1024x1024'.

        Args:
            args_dict (Dict): 

        Returns:
            RegistrationOptions: 
        """
        opts = RegistrationOptions()
        for key in args_dict:
            if key in ['affine_registration_options', 
                       'nonrigid_registration_options', 
                       'segmentation',
                       'tiling_options']:
                continue
            opts.__assign_if_present(key, args_dict)
        # TODO: Remove that.
        if 'affine_registration_options' in  args_dict:
            opts.affine_registration_options.parse_dict(args_dict['affine_registration_options'])
        if 'nonrigid_registration_options' in args_dict:
            opts.nonrigid_registration_options.parse_dict(args_dict['nonrigid_registration_options'])
        if 'segmentation' in args_dict:
            segmentation_options = parse_segmentation_options(args_dict['segmentation'])
            opts.segmentation = segmentation_options
        if 'tiling_options' in args_dict:
            opts.tiling_options.parse_dict(args_dict['tiling_options'])
        return opts

    @staticmethod
    def default_options() -> RegistrationOptions:
        """Get default registration options.

        Returns:
            RegistrationOptions:
        """
        return RegistrationOptions()
    
    @staticmethod
    def affine_only_options() -> RegistrationOptions:
        """Default options for affine registration only.
        
        Returns:
            Registration Options:
        """
        opts = RegistrationOptions()
        opts.do_nonrigid_registration = False
        return opts
    
    @staticmethod
    def nonrigid_only_options() -> RegistrationOptions:
        """Default options for nonrigid registration only.

        Returns:
            RegistrationOptions: 
        """
        opts = RegistrationOptions()
        opts.do_affine_registration = False
        opts.disable_mask_generation = True
        return opts
    
    @staticmethod
    def nrpt_only_options() -> RegistrationOptions:
        """Default options for nrpt registration only.

        Returns:
            RegistrationOptions: 
        """
        opts = RegistrationOptions()
        opts.do_affine_registration = False
        return opts