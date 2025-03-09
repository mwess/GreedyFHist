from dataclasses import dataclass


def parse_segmentation_options(config: dict):
    segmentation_class = config.get('segmentation_class', 'YoloSegOptions')
    if segmentation_class == 'YoloSegOptions':
        options = YoloSegOptions()
    elif segmentation_class == 'TissueEntropySegOptions':
        options = TissueEntropySegOptions()
    elif segmentation_class == 'LuminosityAndAreaSegOptions':
        options = LuminosityAndAreaSegOptions()
    else:
        raise Exception(f'Segmentation class unkown: {segmentation_class}')
    options.parse_dict(config)
    return options


@dataclass
class SegmentationOptions:
    pass
    
    def parse_dict(self, args_dict: dict):
        """Function made to automatically parse attributes from dictionary. 

        Args:
            args_dict (Dict): 
        """
        for key in args_dict:
            self.__assign_if_present(key, args_dict)

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

@dataclass
class TissueEntropySegOptions(SegmentationOptions):
    """Options for configuring tissue detection using entropy.

    Attributes:
        target_resolution: int = 640
            Scale image down so that maximum image dimension corresponds to target_resolution. 
            Defaults to 640.
            
        do_clahe: bool = True
             Uses clahe for contrast enhancement. Defaults to True.
        
        use_luminosity: bool = False
            If True, uses only the luminosity of the LAB channels for computing entropy.
            If this options is used, `convert_to_xyz` is ignored. Defaults to False.
            
        footprint_size: int = 10
            Size of footprint used for entropy and morphological 
            closing. Footprint is of square/cubic shape. Defaults to 10.
            
        convert_to_xyz: bool = False
            Convert image to XYZ image space. Useful if color has spilled outside 
            of the image. Defaults to False.
            
        normalize_entropy: bool = False
            Normalizes entropy so that the max value is 1. Defaults to False.
            
        pre_gaussian_sigma: float = 0.5
            Gaussian filter applied before applying Otsu. Defaults to 0.5.
            
        area_opening_connectivity: int = 1
            Connectivity for removing small objects.
            If None, no opening is applied. Defaults to 1.
            
        area_opening_threshold: int = 100
            Minimum area threshold for removing small objects. 
            If None, no opening is applied. Defaults to 100.
            
        post_gaussian_sigma: float = 0.5
            Gaussian filtering applied after morphological opening. Defaults to 0.5.
            
        with_morphological_closing: bool = True
            Performs morphological closing to connect bits of mask. Defaults to True.
            
        do_fill_hole: bool = True
            Fills holes. Defaults to True.
    """
    
    target_resolution: int = 640,
    do_clahe: bool = True,    
    use_luminosity: bool = False,                                              
    footprint_size: int = 10, 
    convert_to_xyz: bool = False,
    normalize_entropy: bool = False,                         
    pre_gaussian_sigma: float = 0.5,
    area_opening_connectivity: int = 1,
    area_opening_threshold: int = 100,
    post_gaussian_sigma: float = 0.5,
    with_morphological_closing: bool = True,
    do_fill_hole: bool = True

    @staticmethod
    def default_options() -> SegmentationOptions:
        return TissueEntropySegOptions()
    

@dataclass
class LuminosityAndAreaSegOptions(SegmentationOptions):
    """Options for the luminosity and area segmentation function.
    
    Attributes:
        target_resolution: int = 640
            Downscaling of the image's maximum dimension. Defaults to 640.
            
        disk_size: int = 1
            Size of disk used for morphological operations. Defaults to 1.
            
        with_morphological_erosion: bool = True
            Enables morphological erosion.
            
        with_morphological_closing: bool = False
            Enables morphological closing.
            
        min_area_size: int = 100
            Threshold to distinguish small and big areas. Defaults to 100.
            
        distance_threshold: int = 30
            Threshold for removing small distant areas. Defaults to 30.
            
        low_intensity_rem_threshold: int = 25
            Removes low intensity threshold. Defaults to 25.
            
        with_hole_filling: bool = True
            Fills holes.
    """
    target_resolution: int = 640
    disk_size: int = 1
    with_morphological_erosion: bool = True
    with_morphological_closing: bool = False
    min_area_size: int = 100
    distance_threshold: int = 30
    low_intensity_rem_threshold: int = 25
    with_hole_filling: bool = True
    
    @staticmethod
    def default_options() -> SegmentationOptions:
        return LuminosityAndAreaSegOptions()
    

@dataclass
class YoloSegOptions(SegmentationOptions):
    """
    Options for configuring YOLO8 based tissue segmentation.
    
    Attributes:
    
        min_area_size: int = 10000
            Remove artifacts after segmentation smaller than threshold.
            
        use_tv_chambolle: bool = True
            Remove noise using total variation minimization by Chambolle.
            
        use_clahe: bool = False
            Enhance contrasts using clahe.
            
        fill_holes: bool = True
            If True, fills holes after segmentation.
            
        use_fallback: str | None = None
            Either 'otsu' or None at the moment. If 'otsu' applies Otsu thresholding if the the Yolo segmentation
            fails. If None and Yolo segmentation fails, returns a full mask.
    """
    min_area_size: int = 10000
    use_tv_chambolle: bool = True
    use_clahe: bool = False
    fill_holes: bool = True
    use_fallback: str | None = None
    
    @staticmethod
    def default_options() -> SegmentationOptions:
        return YoloSegOptions()