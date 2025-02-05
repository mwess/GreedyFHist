"""
This module handles the 
"""

from os.path import join, split
from typing import Callable

import cv2
import numpy, numpy as np
from skimage.measure import regionprops, label
from skimage.restoration import denoise_tv_chambolle
from skimage.morphology import reconstruction
import onnxruntime as ort

from greedyfhist.segmentation.yolo8_parsing import postprocess


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