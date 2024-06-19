import os
from os.path import join, split
from typing import Callable

import cv2
import numpy
import numpy as np
from skimage.measure import regionprops, label
from skimage.restoration import denoise_tv_chambolle
from skimage.morphology import reconstruction
# from ultralytics import YOLO
import onnxruntime as ort

from greedyfhist.segmentation.yolo8_parsing import postprocess


def fill_hole(mask: numpy.array) -> numpy.array:
    """Fills any holes in the computed mask.

    Args:
        mask (numpy.array): 2d mask.

    Returns:
        numpy.array: Mask filled with holes.
    """
    seed = np.copy(mask)
    seed[1:-1, 1:-1] = 1 
    mask_ = mask 

    filled = reconstruction(seed, mask_, method='erosion')
    return filled


def preprocess_for_segmentation(image: numpy.array) -> numpy.array:
    """Applies preprocessing steps to image before segmentation:
        1. Downscaling (640x640)
        2. Grayscale conversion
        3. Soft denoising.
        4. Stacking of grayscale images.

    Args:
        image (numpy.array): Image to preprocess

    Returns:
        numpy.array: Preprocessed image.
    """
    image = cv2.resize(image, (640, 640))
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img2 = denoise_tv_chambolle(image_gray, weight=0.1, channel_axis=-1)
    img2 = (img2 * 255).astype(np.uint8)
    preprocessed_image = np.stack((img2, img2, img2))
    preprocessed_image = np.moveaxis(preprocessed_image, 0, 2)
    return preprocessed_image


def resolve_path_to_model() -> str:
    """Returns path to YOLO8 segmentation model.

    Returns:
        str: Path to model.
    """
    current_dir = split(__file__)[0]
    model_path = join(split(current_dir[:-1])[0], 'model_data/segmentation/default_yolo.onnx')
    return model_path
    


def load_yolo_segmentation() -> Callable:
    """Loads YOLO8 based segmentation function.

    Returns:
        Callable: Segmentation function.
    """

    model_path = resolve_path_to_model()
    ort_session = ort.InferenceSession(model_path)
    output_names = [x.name for x in ort_session.get_outputs()]
    input_name = [x.name for x in ort_session.get_inputs()]
    IMAGE_SHAPE = (640, 640)


    def _predict(image: numpy.array, min_area_size: int = 10000, fill_holes: bool = True) -> numpy.array:
        """Segmentation function for foreground segmentation of histology image.

        Args:
            image (numpy.array): Image to segment.
            min_area_size (int, optional): Filters out all smaller patches of misclassified noise by removing every region smaller than limit. Defaults to 10000.
            fill_holes (bool, optional): If True, fills all holes in mask.. Defaults to True.

        Returns:
            numpy.array: Segmented image.
        """
        preprocessed_image = preprocess_for_segmentation(image)
        preprocessed_image = (np.expand_dims(np.moveaxis(preprocessed_image, 2, 0), 0) / 255.).astype(np.float32)

        outputs = ort_session.run(output_names, {'images': preprocessed_image}, None)
        _, _, prediction = postprocess(outputs, shape=IMAGE_SHAPE)
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