"""
Contains functionality to segment images
"""
import os
from os.path import join, split

import cv2
import numpy
import numpy as np
from skimage.measure import regionprops, label
from skimage.restoration import denoise_tv_chambolle
# from ultralytics import YOLO
import onnxruntime as ort

from greedyfhist.segmentation.yolo8_parsing import postprocess_segmentation

path_to_model = 'data/model/model.onnx'


def preprocess_for_segmentation(image: numpy.array):
    image = cv2.resize(image, (640, 640))
    image_gray = cv2.cvtColor(image, cv2.RGB2GRAY)
    img2 = denoise_tv_chambolle(image_gray, weight=0.1, channel_axis=-1)
    img2 = (img2 * 255).astype(np.uint8)
    preprocessed_image = np.stack((img2, img2, img2))
    return preprocessed_image

def preprocess_for_segmentation2(image: numpy.array):
    image = cv2.resize(image, (640, 640))
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img2 = denoise_tv_chambolle(image_gray, weight=0.1, channel_axis=-1)
    img2 = (img2 * 255).astype(np.uint8)
    preprocessed_image = np.stack((img2, img2, img2))
    preprocessed_image = np.moveaxis(preprocessed_image, 0, 2)
    return preprocessed_image

# def segment_image(image: numpy.array, min_area_size: int = 500):
#     preprocessed_image = preprocess_for_segmentation(image)
#     model = YOLO(path_to_model)
#     prediction = model(preprocessed_image)
#     pred = prediction[0]
#     mask = pred.masks.data.numpy()
#     if len(mask.shape) > 2:
#         print('Warning! More than one mask predicted. Using only the first part for now.')
#         mask = mask[0]
#     filtered_mask = np.zeros_like(mask)
#     # Filter out small regions.
#     regions = regionprops(label(mask))
#     for region in regions:
#         if region.area > min_area_size:
#             minr, minc, maxr, maxc = region.bbox
#             filtered_mask[minr:maxr, minc:maxc] = region.image.astype(np.uint8)
#     return filtered_mask
#     mask = mask.squeeze()
#     # TODO: Make sure that order of dimensions is correct
#     # Add support for filtering smaller objects
#     return cv2.resize(mask, (image.shape[1], image.shape[0]))


def preprocess_for_segmentation(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (640, 640))
    image = np.dstack((image, image, image))
    return image


def resolve_path_to_model():
    current_dir = split(__file__)[0]
    model_path = join(split(current_dir[:-1])[0], 'model_data/segmentation/default_yolo.onnx')
    return model_path
    

# def load_yolo_segmentation2():

#     # model_path = 'intern_data/segmentation/default_yolo.onnx'
#     model_path = resolve_path_to_model()
#     model = YOLO(model_path, task='segment')

#     def _predict(image, min_area_size: int = 20000):
#         preprocessed_image = preprocess_for_segmentation2(image)
#         prediction = model(preprocessed_image, task='segment')
#         mask = prediction[0].masks.data.numpy()
#         if len(mask.shape) > 2:
#             # print('Warning! More than one mask predicted. Using only the first part for now.')
#             mask = mask[0]
#         filtered_mask = np.zeros_like(mask)
#         # Filter out small regions.
#         regions = regionprops(label(mask))
#         for region in regions:
#             if region.area > min_area_size:
#                 minr, minc, maxr, maxc = region.bbox
#                 filtered_mask[minr:maxr, minc:maxc] = region.image.astype(np.uint8)
#         filtered_mask = cv2.resize(filtered_mask, (image.shape[1], image.shape[0]))
#         filtered_mask[filtered_mask != 0] = 1
#         filtered_mask = filtered_mask.astype(np.uint8)
#         # filtered_mask = cv2.resize(filtered_mask, (image.shape[1], image.shape[0]))
#         return filtered_mask
#     return _predict
    

def load_yolo_segmentation():

    # model_path = 'intern_data/segmentation/default_yolo.onnx'
    model_path = resolve_path_to_model()
    # model = YOLO(model_path, task='segment')
    ort_session = ort.InferenceSession(model_path)
    output_names = [x.name for x in ort_session.get_outputs()]
    input_name = [x.name for x in ort_session.get_inputs()]
    IMAGE_SHAPE = (640, 640)


    def _predict(image, min_area_size: int = 500):
        preprocessed_image = preprocess_for_segmentation2(image)
        preprocessed_image = (np.expand_dims(np.moveaxis(preprocessed_image, 2, 0), 0) / 255.).astype(np.float32)

        outputs = ort_session.run(output_names, {'images': preprocessed_image}, None)
        # prediction = model(preprocessed_image, task='segment')
        prediction = postprocess_segmentation(outputs, IMAGE_SHAPE)
        # mask = prediction[0].masks.data.numpy()
        mask = prediction[0]
        if len(mask.shape) > 2:
            # print('Warning! More than one mask predicted. Using only the first part for now.')
            mask = np.squeeze(mask)
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
        # filtered_mask = cv2.resize(filtered_mask, (image.shape[1], image.shape[0]))
        return filtered_mask
    return _predict
    
