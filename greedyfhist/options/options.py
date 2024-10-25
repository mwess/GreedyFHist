from dataclasses import dataclass, field
from typing import Any

def get_3_step_pyramid_iterations():
    return [100, 50, 10]


def get_4_step_pyramid_iterations():
    return [100, 100, 50, 10]


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

    yolo_segmentation_min_size: int
        Threshold for recognition of tissue. Everything smaller is removed from masks.        
    """

    moving_sr: int = 30
    moving_sp: int = 25
    fixed_sr: int = 30
    fixed_sp: int = 25
    yolo_segmentation_min_size: int = 5000
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
            
        rigid_iterations: Union[int, str]
            Number of rigid iterations during initial registration of affine registration.
            If set to 'auto', the number of iterations for rigid matching
            is computed relative to the size of the offset from the 
            center-of-mass initialization. Otherwise, uses the provided
            numer. 
            
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
    resolution: tuple[int, int] = field(default_factory=load_default_resolution)
    preprocessing_options: 'PreprocessingOptions' = field(default_factory=PreprocessingOptions.default_options)
    kernel_size: int = 10
    cost_function: str = 'ncc'
    rigid_iterations: int | str = 10000
    ia: str = 'ia-com-init'
    iteration_pyramid: list[int] = field(default_factory=get_3_step_pyramid_iterations)
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
    """

    dim: int = 2
    resolution: tuple[int, int] = field(default_factory=load_default_resolution)
    preprocessing_options: PreprocessingOptions = field(default_factory=PreprocessingOptions.default_options)    
    s1: float = 5.0
    s2: float = 5.0
    kernel_size: int = 10
    cost_function: str = 'ncc'
    ia: str = 'ia-com-init'
    iteration_pyramid: list[int] = field(default_factory=get_4_step_pyramid_iterations)
    n_threads: int = 1
    use_sv: bool = False
    use_svlb: bool = False
    exp: int | None = None

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

@dataclass
class RegistrationOptions:
    """
    Contains all the options that can be used to register a moving to a fixed image.
    
    affine_registration_options: AffineRegistrationOptions
        Options for preprocessing and calling Greedy with affine
        registration otions.

    nonrigid_registration_options: NonrigidRegistrationOptions
        Options for preprocessing and calling Greedy with nonrigid
        registration options.

    pre_sampling_factor: Union[float, str] = 'auto'
        Sampling factor prior to preprocessing. Does not
        affect registration accuracy, but can help to speed up the 
        registration considerable, especially for large images. If
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

    yolo_segmentation_min_size: int
        Threshold for recognition of tissue. Everything smaller is removed from masks.        
    """

    affine_registration_options: AffineGreedyOptions = field(default_factory=AffineGreedyOptions.default_options)
    nonrigid_registration_options: NonrigidGreedyOptions = field(default_factory=NonrigidGreedyOptions.default_options)
    pre_sampling_factor: float | str = 'auto'
    pre_sampling_max_img_size: int | None = 2000    
    do_affine_registration: bool = True
    do_nonrigid_registration: bool = True
    compute_reverse_nonrigid_registration: bool = False
    temporary_directory: str = 'tmp'
    remove_temporary_directory: bool = True
    yolo_segmentation_min_size: int = 5000
    
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
            args_dict (Dict): _description_

        Returns:
            RegistrationOptions: _description_
        """
        opts = RegistrationOptions()
        for key in args_dict:
            if key in ['affine_registration_options', 'nonrigid_registration_options']:
                continue
            opts.__assign_if_present(key, args_dict)
        # TODO: Remove that.
        if 'affine_registration_options' in  args_dict:
            opts.affine_registration_options.parse_dict(args_dict['affine_registration_options'])
        if 'nonrigid_registration_options' in args_dict:
            opts.nonrigid_registration_options.parse_dict(args_dict['nonrigid_registration_options'])
        return opts

    @staticmethod
    def default_options():
        return RegistrationOptions()