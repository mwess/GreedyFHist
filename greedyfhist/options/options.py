from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union, Dict, Any


def get_3_step_pyramid_iterations():
    return [100, 50, 10]


def get_4_step_pyramid_iterations():
    return [100, 100, 50, 10]


def load_greedyoptions():
    return GreedyOptions()


def load_default_resolution():
    return (1024, 1024)


@dataclass
class PreprocessingOptions:
    """
    Contains all the options that can be used to register a moving to a fixed image.
        
    moving_sr: int = 30
        Color window radius for mean shift filtering in moving image.

    moving_sp: int = 25
        Pixel window readius for mean shift filtering in moving image.
        
    fixed_sr: int = 30
        Color window radius for mean shift filtering in fixed image.
        
    fixed_sp: int = 25
        Pixel window radius for mean shift filtering in fixed image.

    temporary_directory: str = 'tmp'
        Temporary directory used for storing Greedy in- and output.
        
    remove_temporary_directory: bool = True
        Sets whether the temporary directory is removed after
        registration.

    yolo_segmentation_min_size: int
        Threshold for recognition of tissue. Everything smaller is removed from masks.        
    """

    moving_sr: int = 30
    moving_sp: int = 25
    fixed_sr: int = 30
    fixed_sp: int = 25
    temporary_directory: str = 'tmp'
    remove_temporary_directory: bool = True
    yolo_segmentation_min_size: int = 5000
    enable_denoising: bool = True

    
    # def __post_init__(self):
    #     self.greedy_opts = GreedyOptions()
    
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
            if isinstance(value, PreprocessingOptions):
                d[key] = value.to_dict()
            else:
                d[key] = value
        return d

    @staticmethod
    def parse_cmdln_dict(args_dict: Dict) -> 'PreprocessingOptions':
        """Sets all values in dictionary that have a matching key in 
        class's __annotation__ field. key/value pairs for GreedyOpts
        are put in a sub dict, called 'greedy'. 'resolution' is in
        string, e.g. '1024x1024'.

        Args:
            args_dict (Dict): _description_

        Returns:
            RegistrationOptions: _description_
        """
        opts = PreprocessingOptions()
        for key in args_dict:
            if key in ['greedy', 'resolution']:
                continue
            opts.__assign_if_present(key, args_dict)
        return opts

    @staticmethod
    def default_options():
        return PreprocessingOptions()
    

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
            
        rigid_iterations: int
            Number of rigid iterations during initial registration of affine registration.
            
        ia: str
            Initial image alignment. 'ia-com-init' is a custom option for using the center-of-mass of image masks. Other options can be taken from Greedy. 
            
        iteration_pyramid: List[int]
            Iterations in the multiresolution pyramid.
            
        n_threads: int
            Number of threads. Defaults to 1, but 8 has shown the fastest registrations.
            
        yolo_segmentation_min_size: int
            Threshold for recognition of tissue. Everything smaller is removed from masks.

        enable_denoising: bool
            Use denoising prior to registration.
        
        keep_affine_transform_unbounded: bool = True
            If true, keeps affine transform unbounded. Otherwise, affine
            transform is translated into a displacement field. Should be
            set to True.            
    """

    dim: int = 2
    resolution: Tuple[int, int] = field(default_factory=load_default_resolution)
    preprocessing_options: 'PreprocessingOptions' = field(default_factory=PreprocessingOptions.default_options)
    kernel_size: int = 10
    cost_function: str = 'ncc'
    rigid_iterations: int = 10000
    ia: str = 'ia-com-init'
    iteration_pyramid: List[int] = field(default_factory=get_3_step_pyramid_iterations)
    n_threads: int = 1
    enable_denoising: bool = True
    keep_affine_transform_unbounded: bool = True

    def __post_init__(self):
        self.preprocessing_options = PreprocessingOptions()

    def parse_dict(self, args_dict: Dict):
        """Function made to automatically parse attributes from dictionary. 

        Args:
            args_dict (Dict): 
        """
        for key in args_dict:
            self.__assign_if_present(key, args_dict)

    def __assign_if_present(self, key: Any, args_dict: Dict):
        """Assigns value if key is present in class's __annotations__.

        Args:
            key (Any):
            args_dict (Dict):
        """
        if key in args_dict:
            value = args_dict[key]
            if key in self.__annotations__:
                self.__setattr__(key, value)

    def to_dict(self) -> Dict:
        """Return options as dictionary.

        Returns:
            Dict:
        """
        d = {}
        for key in self.__annotations__:
            d[key] = self.__getattribute__(key)
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

        enable_denoising: bool
            Use denoising prior to registration.            
    """

    dim: int = 2
    resolution: Tuple[int, int] = field(default_factory=load_default_resolution)
    preprocessing_options: PreprocessingOptions = field(default_factory=PreprocessingOptions.default_options)    
    s1: float = 5.0
    s2: float = 5.0
    kernel_size: int = 10
    cost_function: str = 'ncc'
    ia: str = 'ia-com-init'
    iteration_pyramid: List[int] = field(default_factory=get_4_step_pyramid_iterations)
    n_threads: int = 1
    use_sv: bool = False
    use_svlb: bool = False
    exp: Optional[int] = None

    def __post_init__(self):
        self.preprocessing_options.enable_denoising = False

    def parse_dict(self, args_dict: Dict):
        """Function made to automatically parse attributes from dictionary. 

        Args:
            args_dict (Dict): 
        """
        for key in args_dict:
            self.__assign_if_present(key, args_dict)

    def __assign_if_present(self, key: Any, args_dict: Dict):
        """Assigns value if key is present in class's __annotations__.

        Args:
            key (Any):
            args_dict (Dict):
        """
        if key in args_dict:
            value = args_dict[key]
            if key in self.__annotations__:
                self.__setattr__(key, value)

    def to_dict(self) -> Dict:
        """Return options as dictionary.

        Returns:
            Dict:
        """
        d = {}
        for key in self.__annotations__:
            d[key] = self.__getattribute__(key)
        return d
            
    @staticmethod
    def default_options() -> 'NonrigidGreedyOptions':
        """Returns default options.

        Returns:
            GreedyOptions:
        """
        return NonrigidGreedyOptions()

@dataclass
class RegistrationOptions2:
    """
    Contains all the options that can be used to register a moving to a fixed image.
    
    affine_registration_options: AffineRegistrationOptions
        Options for preprocessing and calling Greedy with affine
        registration otions.

    nonrigid_registration_options: NonrigidRegistrationOptions
        Options for preprocessing and calling Greedy with nonrigid
        registration options.

    pre_sampling_factor: Union[float, str] = 1
        Sampling factor prior to preprocessing. Does not
        affect registration accuracy, but can help to speed up the 
        registration considerable, especially for large images. If
        the factor is a float it is interpreted as a scaling factor
        that is applied on both images prior to preprocessing. If
        'auto', then the pre_sampling_auto_factor is used to
        scale both images to have a maximum resolution of that factor.
        
    pre_sampling_auto_factor: Optional[int] = 3500
        Determines the maximum resolution if pre_sampling_factor is
        set to 'auto'. Ignored otherwise.
        Helpful when image size is unkown or resolution between moving
        and fixed image varies too much.

    do_affine_registration: bool = True
        Whether an affine registration is performed or not.
        
    do_nonrigid_registration: bool = True
        Whether a deformable registration is performed or not.        
        
    keep_affine_transform_unbounded: bool = True
        If true, keeps affine transform unbounded. Otherwise, affine
        transform is translated into a displacement field. Should be
        set to True.
        
    temporary_directory: str = 'tmp'
        Temporary directory used for storing Greedy in- and output.
        
    remove_temporary_directory: bool = True
        Sets whether the temporary directory is removed after
        registration.

    yolo_segmentation_min_size: int
        Threshold for recognition of tissue. Everything smaller is removed from masks.        
    """

    affine_registration_options: AffineGreedyOptions = field(default_factory=AffineGreedyOptions.default_options)
    nonrigid_registration_options: NonrigidGreedyOptions = field(default_factory=NonrigidGreedyOptions.default_options)
    pre_sampling_factor: Union[float, str] = 1
    pre_sampling_auto_factor: Optional[int] = 3500    
    do_affine_registration: bool = True
    do_nonrigid_registration: bool = True
    temporary_directory: str = 'tmp'
    remove_temporary_directory: bool = True
   # TODO: Parse this option correctly.
    yolo_segmentation_min_size: int = 5000
    
    # def __post_init__(self):
    #     self.greedy_opts = ()
    
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
            if isinstance(value, GreedyOptions):
                d[key] = value.to_dict()
            else:
                d[key] = value
        return d

    @staticmethod
    def parse_cmdln_dict(args_dict: Dict) -> 'RegistrationOptions':
        """Sets all values in dictionary that have a matching key in 
        class's __annotation__ field. key/value pairs for GreedyOpts
        are put in a sub dict, called 'greedy'. 'resolution' is in
        string, e.g. '1024x1024'.

        Args:
            args_dict (Dict): _description_

        Returns:
            RegistrationOptions: _description_
        """
        opts = RegistrationOptions()
        for key in args_dict:
            if key in ['greedy', 'resolution']:
                continue
            opts.__assign_if_present(key, args_dict)
        # TODO: Remove that.
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



@dataclass
class GreedyOptions:
    """
    Commandline options parsed to Greedy. 

    More detailed information can be found in Greedy's documentation.
    Some options are extended/overwritten.

    Attributes:
        dim: int 
            Should always be set to 2.
            
        s1: float
            pre sigma value for nonrigid registration.
            
        s2: float
            post sigma value for nonrigid registration.
            
        kernel_size: int
            kernel size for 'ncc' kernel metric.
            
        cost_function: str
            cost function used to optimize the registration. Should always be set to 'ncc'.
            
        rigid_iterations: int
            Number of rigid iterations during initial registration of affine registration.
            
        ia: str
            Initial image alignment. 'ia-com-init' is a custom option for using the center-of-mass of image masks. Other options can be taken from Greedy. 
            
        affine_iteration_pyramid: List[int]
            Iterations in the multiresolution pyramid.
            
        nonrigid_iteration_pyramid: List[int] 
            Iterations in the multiresolution pyramid.
            
        n_threads: int
            Number of threads. Defaults to 1, but 8 has shown the fastest registrations.
            
        use_sv: bool
            Additional greedy option for nonrigid registration.
            
        use_svlb: bool
            Additional experimental greedy option for nonrigid registration.
            
        exp: Optional[int]
            Additional value used in conjunction with use_svlb.

    """

    dim: int = 2
    s1: float = 5.0
    s2: float = 5.0
    kernel_size: int = 10
    cost_function: str = 'ncc'
    rigid_iterations: int = 10000
    ia: str = 'ia-com-init'
    affine_iteration_pyramid: List[int] = field(default_factory=get_3_step_pyramid_iterations)
    nonrigid_iteration_pyramid: List[int] = field(default_factory=get_4_step_pyramid_iterations)
    n_threads: int = 1
    use_sv: bool = False
    use_svlb: bool = False
    exp: Optional[int] = None
    # TODO: Parse this option correctly.
    yolo_segmentation_min_size: int =5000

    def parse_dict(self, args_dict: Dict):
        """Function made to automatically parse attributes from dictionary. 

        Args:
            args_dict (Dict): 
        """
        for key in args_dict:
            self.__assign_if_present(key, args_dict)

    def __assign_if_present(self, key: Any, args_dict: Dict):
        """Assigns value if key is present in class's __annotations__.

        Args:
            key (Any):
            args_dict (Dict):
        """
        if key in args_dict:
            value = args_dict[key]
            if key in self.__annotations__:
                self.__setattr__(key, value)

    def to_dict(self) -> Dict:
        """Return options as dictionary.

        Returns:
            Dict:
        """
        d = {}
        for key in self.__annotations__:
            d[key] = self.__getattribute__(key)
        return d
            
    @staticmethod
    def default_options() -> 'GreedyOptions':
        """Returns default options.

        Returns:
            GreedyOptions:
        """
        return GreedyOptions()




@dataclass
class RegistrationOptions:
    """
    Contains all the options that can be used to register a moving to a fixed image.
    
    greedy_opts: 'GreedyOptions'
        Contains options that are affecting Greedy directly.
        
    resolution: Tuple[int, int]
        Image resolution after downscaling for Greedy.

    do_affine_registration: bool = True
        Whether an affine registration is performed or not.
        
    do_nonrigid_registration: bool = True
        Whether a deformable registration is performed or not.
        
    enable_affine_denoising: bool = True
        Whether images are denoised prior to affine registration.
        
    enable_deformable_denoising: bool = True
        Whether images are denoised prior to nonrigid registration.
        
    moving_sr: int = 30
        Color window radius for mean shift filtering in moving image.

    moving_sp: int = 25
        Pixel window readius for mean shift filtering in moving image.
        
    fixed_sr: int = 30
        Color window radius for mean shift filtering in fixed image.
        
    fixed_sp: int = 25
        Pixel window radius for mean shift filtering in fixed image.
        
    pre_sampling_factor: Union[float, str] = 1
        Sampling factor prior to preprocessing. Does not
        affect registration accuracy, but can help to speed up the 
        registration considerable, especially for large images. If
        the factor is a float it is interpreted as a scaling factor
        that is applied on both images prior to preprocessing. If
        'auto', then the pre_sampling_auto_factor is used to
        scale both images to have a maximum resolution of that factor.
        
    pre_sampling_auto_factor: Optional[int] = 3500
        Determines the maximum resolution if pre_sampling_factor is
        set to 'auto'. Ignored otherwise.
        Helpful when image size is unkown or resolution between moving
        and fixed image varies too much.
        
    keep_affine_transform_unbounded: bool = True
        If true, keeps affine transform unbounded. Otherwise, affine
        transform is translated into a displacement field. Should be
        set to True.
        
    temporary_directory: str = 'tmp'
        Temporary directory used for storing Greedy in- and output.
        
    remove_temporary_directory: bool = True
        Sets whether the temporary directory is removed after
        registration.

    yolo_segmentation_min_size: int
        Threshold for recognition of tissue. Everything smaller is removed from masks.        
    """

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
    pre_sampling_factor: Union[float, str] = 1
    pre_sampling_auto_factor: Optional[int] = 3500
    keep_affine_transform_unbounded: bool = True
    temporary_directory: str = 'tmp'
    remove_temporary_directory: bool = True
    yolo_segmentation_min_size: int = 5000
    
    def __post_init__(self):
        self.greedy_opts = GreedyOptions()
    
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
            if isinstance(value, GreedyOptions):
                d[key] = value.to_dict()
            else:
                d[key] = value
        return d

    @staticmethod
    def parse_cmdln_dict(args_dict: Dict) -> 'RegistrationOptions':
        """Sets all values in dictionary that have a matching key in 
        class's __annotation__ field. key/value pairs for GreedyOpts
        are put in a sub dict, called 'greedy'. 'resolution' is in
        string, e.g. '1024x1024'.

        Args:
            args_dict (Dict): _description_

        Returns:
            RegistrationOptions: _description_
        """
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