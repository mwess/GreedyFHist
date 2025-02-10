"""
This module handles tissue segmentation functions.
"""

from functools import partial
from os.path import join, split
from typing import Callable

import cv2
import numpy, numpy as np
from skimage.filters import gaussian, threshold_otsu
from skimage.filters.rank import entropy
from skimage.measure import regionprops, label
from skimage.morphology import (
    reconstruction, 
    disk, 
    closing, 
    erosion,
    area_opening
)
from skimage.restoration import denoise_tv_chambolle
import onnxruntime as ort

from greedyfhist.segmentation.yolo8_parsing import postprocess
from greedyfhist.options import (
    SegmentationOptions,
    YoloSegOptions,
    TissueEntropySegOptions,
    LuminosityAndAreaSegOptions
)
from greedyfhist.utils.image import scale_image_to_max_dim


def load_segmentation_function(options: SegmentationOptions | Callable[[numpy.ndarray], numpy.ndarray] | str | None = None) -> Callable[[numpy.ndarray], numpy.ndarray]:
    """Loads a segmentation function for detecting tissue segmentation. `options` can be the following: 
    
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

    Args:
        options (SegmentationOptions | Callable[[numpy.ndarray], numpy.ndarray] | str | None, optional): Defaults to None.

    Raises:
        Exception: If unknown str options is passed.
        Exception: If unknown segmentation options object is passed.

    Returns:
        Callable[[numpy.ndarray], numpy.ndarray]: Segmentation function.
    """
    if isinstance(options, Callable):
        return options
    if isinstance(options, str):
        if options == 'yolo-seg':
            return load_yolo_segmentation()
        elif options == 'entropy-based-seg':
            return load_tissue_entropy_detection()
        elif options == 'lum-area-seg':
            return load_tissue_luminosity_area_detection()
        else:
            raise Exception(f'Uknown string command suuplied for segmentation function: {options}.')
    if isinstance(options, SegmentationOptions):
        if isinstance(options, YoloSegOptions):
            return load_yolo_segmentation(
                min_area_size=options.min_area_size,
                fill_holes=options.fill_holes,
                use_tv_chambolle=options.use_tv_chambolle,
                use_clahe=options.use_clahe,
                use_fallback=options.use_fallback
            )
        elif isinstance(options, TissueEntropySegOptions):
            return load_tissue_entropy_detection(
                target_resolution=options.target_resolution,
                do_clahe=options.do_clahe,
                use_luminosity=options.use_luminosity,
                footprint_size=options.footprint_size,
                convert_to_xyz=options.convert_to_xyz,
                normalize_entropy=options.normalize_entropy,
                pre_gaussian_sigma=options.pre_gaussian_sigma,
                area_opening_connectivity=options.area_opening_connectivity,
                area_opening_threshold=options.area_opening_threshold,
                post_gaussian_sigma=options.post_gaussian_sigma,
                with_morphological_closing=options.with_morphological_closing,
                do_fill_hole=options.do_fill_hole
            )
        elif isinstance(options, LuminosityAndAreaSegOptions):
            return load_tissue_luminosity_area_detection(
                target_resolution=options.target_resolution,
                min_area_size=options.min_area_size,
                distance_threshold=options.distance_threshold,
                low_intensity_rem_threshold=options.low_intensity_rem_threshold
            )
        else:
            raise Exception(f'Unkown segmentation option passed: {options}')
    if options is None:
        return load_yolo_segmentation()


def fill_hole(mask: numpy.ndarray) -> numpy.ndarray:
    """Fills any holes in the computed mask.

    Args:
        mask (numpy.ndarray): 2d mask.

    Returns:
        numpy.ndarray: Mask filled with holes.
    """
    seed = np.copy(mask)
    seed[1:-1, 1:-1] = 1 
    mask_ = mask 

    filled = reconstruction(seed, mask_, method='erosion')
    return filled


def _preprocess_for_segmentation(image: numpy.ndarray,
                                use_tv_chambolle: bool = True,
                                use_clahe: bool = False) -> numpy.ndarray:
    """Applies preprocessing steps to image before segmentation:
        1. Downscaling (640x640)
        2. Grayscale conversion
        3. Soft denoising.
        4. Stacking of grayscale images.

    Args:
        image (numpy.ndarray): Image to preprocess

    Returns:
        numpy.ndarray: Preprocessed image.
    """
    image = cv2.resize(image, (640, 640))
    if len(image.shape) == 2:
        image_gray = image
    else:
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img2 = image_gray.copy()
    if use_tv_chambolle:
        img2 = denoise_tv_chambolle(img2, weight=0.1, channel_axis=-1)
        img2 = (img2 * 255).astype(np.uint8)
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img2 = clahe.apply(img2)
    preprocessed_image = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    return preprocessed_image


def _resolve_path_to_model() -> str:
    """Returns path to YOLO8 segmentation model.

    Returns:
        str: Path to model.
    """
    current_dir = split(__file__)[0]
    model_path = join(split(current_dir[:-1])[0], 'model_data/segmentation/default_yolo.onnx')
    return model_path
    


def load_yolo_segmentation(min_area_size: int = 10000,
                           fill_holes: bool = True,
                           use_tv_chambolle: bool = True,
                           use_clahe: bool = False,
                           use_fallback: str | None = 'otsu') -> Callable[[numpy.ndarray], numpy.ndarray]:
    """Loads YOLO8 based segmentation function. We use a closure to initialize the function
    because we cannot parse any additional arguments during the registration itself. All
    arguments are set as constants in the generated prediction function.

    Args:
        min_area_size (int): Threshold for removing artifacts after tissue prediction. Defaults to 10000.
        fill_holes (bool): If True, fill any holes after prediction.
        use_tv_chambolle (bool): If True, uses total variation denoising. Defaults to True.
        use_clahe (bool): If True, uses Contrast Limited Adaptive Histogram Equalization. Defaults to False.
        use_fallback (str, optional): If use_fallback == 'otsu', uses Otsu thresholding on the preprocessed image
            if the YOLO model cannot identify any tissue.

    Returns:
        Callable: Segmentation function.
    """

    model_path = _resolve_path_to_model()
    ort_session = ort.InferenceSession(model_path)
    # ort_session = WrapInferenceSession(model_path)
    output_names = [x.name for x in ort_session.get_outputs()]
    input_name = [x.name for x in ort_session.get_inputs()]
    IMAGE_SHAPE = (640, 640)


    def _predict(image: numpy.ndarray) -> numpy.ndarray:
        """Segmentation function for foreground segmentation of histology image.

        Args:
            image (numpy.ndarray): Image to segment.
            min_area_size (int, optional): Filters out all smaller patches of misclassified noise by removing every region smaller than limit. Defaults to 10000.
            fill_holes (bool, optional): If True, fills all holes in mask.. Defaults to True.

        Returns:
            numpy.ndarray: Segmented image.
        """
        preprocessed_image = _preprocess_for_segmentation(image,
                                                         use_tv_chambolle=use_tv_chambolle,
                                                         use_clahe=use_clahe)
        downscaled_shape = preprocessed_image.shape[:2]
        preprocessed_image2 = (np.expand_dims(np.moveaxis(preprocessed_image, 2, 0), 0) / 255.).astype(np.float32)

        outputs = ort_session.run(output_names, {'images': preprocessed_image2}, None)
        _, _, prediction = postprocess(outputs, shape=IMAGE_SHAPE)
        if isinstance(prediction, list):
            if use_fallback is None:
                prediction = np.expand_dims(np.ones(downscaled_shape).astype(np.uint8), 0)
            elif use_fallback == 'otsu':
                prediction = predict_otsu(preprocessed_image[:,:,0],
                                          use_tv_chambolle=use_tv_chambolle,
                                          use_clahe=use_clahe)
                prediction = np.expand_dims(prediction, 0).astype(np.uint8)
            else:
                raise Exception(f'Fallback method unknown: {use_fallback}.')
        prediction = prediction.astype(np.uint8)
        mask = prediction[0]
        if len(mask.shape) > 2:
            mask = np.squeeze(mask)
        if fill_holes:
            mask = fill_hole(mask)
        filtered_mask = np.zeros_like(mask)
        # Filter out small regions.
        regions = regionprops(label(mask))
        for region in regions:
            if region.area > min_area_size:
                minr, minc, maxr, maxc = region.bbox
                filtered_mask[minr:maxr, minc:maxc] = region.image.astype(np.uint8)
        filtered_mask = cv2.resize(filtered_mask, (image.shape[1], image.shape[0]))
        filtered_mask[filtered_mask != 0] = 1
        filtered_mask = filtered_mask.astype(np.uint8)
        return filtered_mask
    return _predict


def _otsu_thresholding(img: numpy.ndarray):
    blur = cv2.GaussianBlur(img,(5,5),0)
    # find normalized_histogram, and its cumulative distribution function
    hist = cv2.calcHist([blur],[0],None,[256],[0,256])
    hist_norm = hist.ravel()/hist.sum()
    Q = hist_norm.cumsum()
    bins = np.arange(256)
    fn_min = np.inf
    thresh = -1
    for i in range(1,256):
        p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
        q1,q2 = Q[i],Q[255]-Q[i] # cum sum of classes
        if q1 < 1.e-6 or q2 < 1.e-6:
            continue
        b1,b2 = np.hsplit(bins,[i]) # weights
        # finding means and variances
        m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
        v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2
        # calculates the minimization function
        fn = v1*q1 + v2*q2
        if fn < fn_min:
            fn_min = fn
            thresh = i
    _, otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) 
    return otsu


def predict_otsu(image, 
                 use_tv_chambolle: bool = True,
                 use_clahe: bool = False,
                 ):
    """
    Implementation of Otsu tresholding method. Typically not used, but might be useful as a fallback.
    """
    if len(image.shape) == 2:
        gray_image = image
    else:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if use_tv_chambolle:
        gray_image = denoise_tv_chambolle(gray_image, weight=0.1, channel_axis=-1)
        gray_image = (gray_image * 255).astype(np.uint8)
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray_image = clahe.apply(gray_image)
    mask = _otsu_thresholding(gray_image)
    mask = _invert_mask(mask)
    return mask


def _invert_mask(img: numpy.ndarray) -> numpy.ndarray:
    """
    Inverts the mask.
    
    Args:
        img (numpy.ndarray)
        
    Returns:
        numpy.ndarray
    """
    img_new = np.zeros_like(img)
    img_new[img == 0] = 1
    return img_new


def load_tissue_luminosity_area_detection(target_resolution: int = 640,
                                        min_area_size: int = 100,
                                        distance_threshold: int = 30,
                                        low_intensity_rem_threshold: int = 25) -> Callable[[numpy.ndarray], numpy.ndarray]:
    """Loads tissue prediction from luminosity and area detection.

    Args:
        target_resolution (int, optional): Defaults to 640.
        min_area_size (int, optional): Defaults to 100.
        distance_threshold (int, optional): Defaults to 30.
        low_intensity_rem_threshold (int, optional): Defaults to 25.

    Returns:
        Callable[[numpy.ndarray], numpy.ndarray]: Segmentation function.
    """
    return partial(predict_tissue_from_luminosity_and_area,
                   target_resolution=target_resolution,
                   min_area_size=min_area_size,
                   distance_threshold=distance_threshold,
                   low_intensity_rem_threshold=low_intensity_rem_threshold)


def predict_tissue_from_luminosity_and_area(image: numpy.ndarray,
                                   target_resolution: int = 640,
                                   disk_size: int = 1,
                                   with_morphological_erosion: bool = True,
                                   with_morphological_closing: bool = False,
                                   min_area_size: int = 100,
                                   distance_threshold: int = 30,
                                   low_intensity_rem_threshold: int = 25,
                                   with_hole_filling: bool = True
                                   ) -> numpy.ndarray:
    """Predicts tissue from luminosity channel. 
    
    Preprocessing:
            1. Extract luminosity from LAB image space.
            2. Performs CLAHE for contrast enhancing.
            3. Performs morphological erosion.
            4. Performs morphological closing.
            5. Performs Otsu thresholding.
            6. Gaussian filtering. 
            7. Fill holes.
            8. Removes low intensity areas.
            
    This form of preprocessing might lead to small scattered artifacts.
    In the next step, we divide images in small and large areas. Small areas that are close 
    to large areas typically belong to the tissue area.
    We then compute for each small area its distance to the closest large area. If the distance 
    is below the given threshold, its characterized as tissue.
            
    
    Args:
        image (numpy.ndarray): 
        target_resolution (int): Downscaling of the image's maximum dimension. Defaults to 640.
        disk_size (int): Size of disk used for morphological operations. Defaults to 1.
        with_morphological_erosion (bool): Enable morphological erosion. Defaults to True.
        with_morphological_closing (bool): Enable morphological closing. Defaults to False.
        min_area_size (int, optional): Threshold to distinguish small and big areas. Defaults to 100.
        distance_threshold (int, optional): Threshold for removign small distant areas. Defaults to 30.
        low_intensity_rem_threshold (int, optional): Removes low intensity threshold. Defaults to 25.
        with_hole_filling (bool): Fills holes.

    Returns:
        numpy.ndarray: _description_
    """
    shape = image.shape[:2]
    img = scale_image_to_max_dim(image, target_resolution)
    mask = preprocessing_for_l_segmentation(img, 
                                            disk_size=disk_size,
                                            with_erosion=with_morphological_erosion,
                                            with_closing=with_morphological_closing,
                                            low_intensity_rem_threshold=low_intensity_rem_threshold, 
                                            with_hole_filling=with_hole_filling)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    small_contours, large_contours = [], []
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area > min_area_size:
            large_contours.append(contour)
        else:
            small_contours.append(contour)
    min_dists = []
    remaining_contours = []
    for i in range(len(small_contours)):
        dists = []
        for j in range(len(large_contours)):
            dist = compute_nearest_distance_between_contours(small_contours[i], large_contours[j])
            if dist < distance_threshold:
                # remaining_contours.append(small_contours[i])
                dists.append(dist)
        if dists:
            min_dists.append(np.array(dists).min())
            remaining_contours.append(small_contours[i])
    min_dists = np.array(min_dists)

    final_contours = large_contours + remaining_contours
    # print(min_dists)            
    template = np.zeros_like(mask)
    new_mask = cv2.fillPoly(template, final_contours, 1)
    new_mask = cv2.resize(new_mask, shape[::-1], cv2.INTER_NEAREST)
    return new_mask


def preprocessing_for_l_segmentation(img: numpy.ndarray, 
                  disk_size: int = 1, 
                  with_erosion: bool = True,
                  with_closing: bool = False,
                  with_hole_filling: bool = True,
                  low_intensity_rem_threshold: int = None) -> numpy.ndarray:
    """Preprocessing for luminosity segmentation.
    
    1. Extract luminosity from LAB image space.
    2. Performs CLAHE for contrast enhancing.
    3. Performs morphological erosion.
    4. Performs morphological closing.
    5. Performs Otsu thresholding.
    6. Gaussian filtering. 
    7. Fill holes.
    8. Removes low intensity areas.

    Args:
        img (numpy.ndarray): 
        disk_size (int, optional): Size of disk for morphological operations. Defaults to 1.
        with_erosion (bool, optional):  Defaults to True.
        with_closing (bool, optional):  Defaults to False.
        with_hole_filling (bool, optional):  Defaults to True.
        low_intensity_rem_threshold (int, optional): Defaults to None.

    Returns:
        numpy.ndarray: 
    """
    # Get LAB
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    sat = img_lab[:,:,0]
    
    # Enhance contrast in luminosity.
    clahe = cv2.createCLAHE(2, (8,8))
    sat = clahe.apply(sat)        
    footprint = disk(disk_size)   
    
    # Low sample areas can be enhanced using morphological erosion.
    if with_erosion:
        sat = erosion(sat, footprint)
    
    # This can be further enhanced by morphological closing.
    if with_closing:
        footprint = disk(disk_size)
        sat = closing(sat, footprint)
        
    # Now we use Otsu to separate background.
    thresh = threshold_otsu(sat)        
    sat_cpy = sat.copy()
    sat_cpy[sat > thresh] = 0
    sat_cpy[sat_cpy > 0] = 255
    sat_cpy = sat_cpy/255.
    
    # With some additional gaussian filtering to capture small holes and connect areas.
    sat_cpy = gaussian(sat_cpy, sigma=2)
    
    # Now we can fill holes if we are primarily interested finding the tissue area.
    if with_hole_filling:
        sat_cpy = fill_hole(sat_cpy)      
    sat_cpy = (sat_cpy *255).astype(np.uint8)
    if low_intensity_rem_threshold is not None:
        sat_cpy[sat_cpy <= low_intensity_rem_threshold] = 0
    mask = sat_cpy.copy()
    mask[mask > 0] = 1
    mask = mask.astype(np.uint8)
    return mask


def compute_nearest_distance_between_contours(cont_a, cont_b) -> float:
    """Computes the nearest distance between two countours.

    Args:
        cont_a: 
        cont_b: 

    Returns:
        float: distance.
    """
    distances = []
    for i in range(len(cont_a)):
        p = tuple(int(x) for x in cont_a[i][0].squeeze())
        dist = -cv2.pointPolygonTest(cont_b, p, True)
        distances.append(dist)
    return np.array(distances).min()


def load_tissue_entropy_detection(target_resolution: int = 640,
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
                         do_fill_hole: bool = True) -> Callable[[numpy.ndarray], numpy.ndarray]:
    """Since we cannot parse segmentation arguments during segmentation we need to load a preconfigured
    loader function.

    Args:
        target_resolution (int): Scale image down so that maximum image dimension corresponds to target_resolution. 
            Defaults to 640.
        do_clahe (bool, optional): Defaults to True.
        use_luminosity (bool, optional): If True, uses only the luminosity of the LAB channels for computing entropy.
            If this options is used, `convert_to_xyz` is ignored. Defaults to False.
        footprint_size (int, optional): Size of footprint used for entropy and morphological 
            closing. Footprint is of square/cubic shape. Defaults to 10.
        convert_to_xyz (bool, optional): Convert image to XYZ image space. Useful if color has spilled outside 
            of the image. Defaults to False.
        normalize_entropy (bool, optional): Normalizes entropy so that the max value is 1. Defaults to False.
        pre_gaussian_sigma (float, optional): Gaussian filter applied before applying Otsu. Defaults to 0.5.
        area_opening_connectivity (int, optional): Connectivity for removing small objects.
            If None, no opening is applied. Defaults to 1.
        area_opening_threshold (int, optional): Minimum area threshold for removing small objects. 
            If None, no opening is applied. Defaults to 100.
        post_gaussian_sigma (float, optional): Gaussian filtering applied after morphological opening. Defaults to 0.5.
        with_morphological_closing (bool, optional): Performs morphological closing to connect bits of mask. Defaults to True.
        do_fill_hole (bool, optional): Fills holes. Defaults to True.

    Returns:
        Callable[[numpy.ndarray], numpy.ndarray]: segmentation function.
    """
    return partial(predict_tissue_from_entropy, 
                   target_resolution=target_resolution,
                   do_clahe=do_clahe,
                   use_luminosity=use_luminosity,
                   footprint_size=footprint_size,
                   convert_to_xyz=convert_to_xyz,
                   normalize_entropy=normalize_entropy,
                   pre_gaussian_sigma=pre_gaussian_sigma,
                   area_opening_connectivity=area_opening_connectivity,
                   area_opening_threshold=area_opening_threshold,
                   post_gaussian_sigma=post_gaussian_sigma,
                   with_morphological_closing=with_morphological_closing,
                   do_fill_hole=do_fill_hole)
    

# TODO: Add option for ignoring black areas after transformation.
def predict_tissue_from_entropy(image: numpy.ndarray, 
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
                         do_fill_hole: bool = True) -> numpy.ndarray:
    """Entropy based tissue segmentation. The base assumption is that tissue areas
    exhibit a higher entropy than non tissue area.
    
    The standard algorithm works as follows:
        1. Downscale image.
        2. Apply CLAHE to enhance contrasts.
        3. Convert image to XYZ.
        4. Compute entropy of image.
        5. Gaussian denoising on entropy.
        6. Average the 3 channels of entropy.
        7. Apply Otsu's thresholding.
        8. Apply morphological opening to remove small artifacts.
        9. Perform Gaussian denoising on computed mask.
        10. Perform morpholohigical closing to connect small open areas.
        11. Fill holes.
        12. Rescale image.

    Args:
        img (numpy.ndarray): 
        target_resolution (int): Scale image down so that maximum image dimension corresponds to target_resolution. 
            Defaults to 640.
        do_clahe (bool, optional): Uses clahe for contrast enhancement. Defaults to True.
        use_luminosity (bool, optional): If True, uses only the luminosity of the LAB channels for computing entropy.
            If this options is used, `convert_to_xyz` is ignored. Defaults to False.
        footprint_size (int, optional): Size of footprint used for entropy and morphological 
            closing. Footprint is of square/cubic shape. Defaults to 10.
        convert_to_xyz (bool, optional): Convert image to XYZ image space. Useful if color has spilled outside 
            of the image. Defaults to False.
        normalize_entropy (bool, optional): Normalizes entropy so that the max value is 1. Defaults to False.
        pre_gaussian_sigma (float, optional): Gaussian filter applied before applying Otsu. Defaults to 0.5.
        area_opening_connectivity (int, optional): Connectivity for removing small objects.
            If None, no opening is applied. Defaults to 1.
        area_opening_threshold (int, optional): Minimum area threshold for removing small objects. 
            If None, no opening is applied. Defaults to 100.
        post_gaussian_sigma (float, optional): Gaussian filtering applied after morphological opening. Defaults to 0.5.
        with_morphological_closing (bool, optional): Performs morphological closing to connect bits of mask. Defaults to True.
        do_fill_hole (bool, optional): Fills holes. Defaults to True.

    Returns:
        numpy.ndarray: 
    """
    # Initial downscaling
    shape = image.shape[:2]
    image = scale_image_to_max_dim(image, target_resolution)    
    # Extract luminosity. TODO: Move that further down.
    img_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lum = img_lab[:,:,2]
    # Constrast enhancing of the luminosity channel.
    if do_clahe:
        clahe = cv2.createCLAHE(2, (8,8))
        lum = clahe.apply(lum)
    if use_luminosity:
        # Compute entropy on L channel.
        footprint = np.ones((footprint_size, footprint_size), dtype=np.uint8)
        ent = entropy(lum, footprint)
    else:
        footprint = np.ones((footprint_size, footprint_size, footprint_size), dtype=np.uint8)
        if do_clahe:
            # If CLAHE is used, convert enhanced image back to RGB.
            img_lab[:,:,2] = lum
            img_ = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
        else:
            img_ = image
        # XYZ color conversion.
        if convert_to_xyz:
            img_ = cv2.cvtColor(img_, cv2.COLOR_RGB2XYZ)            
        ent = entropy(img_, footprint)
        ent = ent.mean(2)
    if normalize_entropy:
        # Normalize entropy
        ent = ent / ent.max()
    if pre_gaussian_sigma is not None:
        # Gaussian filtering before Otsu.
        if use_luminosity:
            channel_axis = None
        else:
            channel_axis = -1
        ent = gaussian(ent, sigma=pre_gaussian_sigma, channel_axis=channel_axis)
    # Otsu thresholding.
    thresh = threshold_otsu(ent)
    mask_ent = (ent > thresh).astype(np.uint8)
    # Morphological area opening.
    if not (area_opening_connectivity is None or area_opening_threshold is None):
        mask_ent = area_opening(mask_ent, 
                                connectivity=area_opening_connectivity, 
                                area_threshold=area_opening_threshold)
    # Post opening Gaussian filtering.
    if post_gaussian_sigma is not None:
        mask_ent = gaussian(mask_ent, sigma=post_gaussian_sigma)
        mask_ent[mask_ent > 0] = 1
    # Morphological closing to make holes from open areas.
    if with_morphological_closing:
        fp = np.ones((footprint_size, footprint_size), dtype=np.uint8)
        mask_ent = closing(mask_ent, fp)
    # Fill holes.
    if do_fill_hole:
        mask_ent = fill_hole(mask_ent)
    mask = mask_ent.astype(np.uint8)
    mask = cv2.resize(mask, shape[::-1], cv2.INTER_NEAREST)
    return mask.astype(np.uint8)