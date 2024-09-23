from typing import Tuple, List, Optional

import cv2
import numpy, numpy as np


def postprocess(preds: Tuple[numpy.array, numpy.array], 
                conf_threshold: float = 0.4, 
                iou_threshold: float = 0.45, 
                nm: int = 32, 
                shape: Tuple[int, int] = (640,640)):
    """
    Post-process the prediction.

    Args:
        preds (Numpy.ndarray): predictions come from ort.session.run().
        im0 (Numpy.ndarray): [h, w, c] original input image.
        ratio (tuple): width, height ratios in letterbox.
        pad_w (float): width padding in letterbox.
        pad_h (float): height padding in letterbox.
        conf_threshold (float): conf threshold.
        iou_threshold (float): iou threshold.
        nm (int): the number of masks.

    Returns:
        boxes (List): list of bounding boxes.
        segments (List): list of segments.
        masks (np.ndarray): [N, H, W], output masks.
    """
    x, protos = preds[0], preds[1]  # Two outputs: predictions and protos

    # Transpose the first output: (Batch_size, xywh_conf_cls_nm, Num_anchors) -> (Batch_size, Num_anchors, xywh_conf_cls_nm)
    x = np.einsum("bcn->bnc", x)

    # Predictions filtering by conf-threshold
    x = x[np.amax(x[..., 4:-nm], axis=-1) > conf_threshold]

    # Create a new matrix which merge these(box, score, cls, nm) into one
    # For more details about `numpy.c_()`: https://numpy.org/doc/1.26/reference/generated/numpy.c_.html
    x = np.c_[x[..., :4], np.amax(x[..., 4:-nm], axis=-1), np.argmax(x[..., 4:-nm], axis=-1), x[..., -nm:]]

    # NMS filtering
    x = x[cv2.dnn.NMSBoxes(x[:, :4], x[:, 4], conf_threshold, iou_threshold)]

    # Decode and return
    if len(x) > 0:
        # Bounding boxes format change: cxcywh -> xyxy
        x[..., [0, 1]] -= x[..., [2, 3]] / 2
        x[..., [2, 3]] += x[..., [0, 1]]

        # Rescales bounding boxes from model shape(model_height, model_width) to the shape of original image
        # x[..., :4] -= [pad_w, pad_h, pad_w, pad_h]
        # x[..., :4] /= min(ratio)

        # Bounding boxes boundary clamp
        x[..., [0, 2]] = x[:, [0, 2]].clip(0, shape[1])
        x[..., [1, 3]] = x[:, [1, 3]].clip(0, shape[0])

        # Process masks
        masks = process_mask(protos[0], x[:, 6:], x[:, :4], shape)

        # Masks -> Segments(contours)
        segments = masks2segments(masks)
        return x[..., :6], segments, masks  # boxes, segments, masks
    else:
        return [], [], []

def masks2segments(masks: numpy.ndarray) -> List[numpy.ndarray]:
    """
    It takes a list of masks(n,h,w) and returns a list of segments(n,xy) (Borrowed from
    https://github.com/ultralytics/ultralytics/blob/465df3024f44fa97d4fad9986530d5a13cdabdca/ultralytics/utils/ops.py#L750)

    Args:
        masks (numpy.ndarray): the output of the model, which is a tensor of shape (batch_size, 160, 160).

    Returns:
        segments (List): list of segment masks.
    """
    segments = []
    for x in masks.astype("uint8"):
        c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]  # CHAIN_APPROX_SIMPLE
        if c:
            c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
        else:
            c = np.zeros((0, 2))  # no segments found
        segments.append(c.astype("float32"))
    return segments

def crop_mask(masks: numpy.ndarray, boxes: numpy.ndarray) -> numpy.ndarray:
    """
    It takes a mask and a bounding box, and returns a mask that is cropped to the bounding box. (Borrowed from
    https://github.com/ultralytics/ultralytics/blob/465df3024f44fa97d4fad9986530d5a13cdabdca/ultralytics/utils/ops.py#L599)

    Args:
        masks (Numpy.ndarray): [n, h, w] tensor of masks.
        boxes (Numpy.ndarray): [n, 4] tensor of bbox coordinates in relative point form.

    Returns:
        (Numpy.ndarray): The masks are being cropped to the bounding box.
    """
    n, h, w = masks.shape
    x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1)
    r = np.arange(w, dtype=x1.dtype)[None, None, :]
    c = np.arange(h, dtype=x1.dtype)[None, :, None]
    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

def process_mask(protos: numpy.ndarray, 
                 masks_in: numpy.ndarray, 
                 bboxes: numpy.ndarray, 
                 im0_shape: Tuple[int, int, int]):
    """
    Takes the output of the mask head, and applies the mask to the bounding boxes. This produces masks of higher quality
    but is slower. (Borrowed from https://github.com/ultralytics/ultralytics/blob/465df3024f44fa97d4fad9986530d5a13cdabdca/ultralytics/utils/ops.py#L618)

    Args:
        protos (numpy.ndarray): [mask_dim, mask_h, mask_w].
        masks_in (numpy.ndarray): [n, mask_dim], n is number of masks after nms.
        bboxes (numpy.ndarray): bboxes re-scaled to original image shape.
        im0_shape (tuple): the size of the input image (h,w,c).

    Returns:
        (numpy.ndarray): The upsampled masks.
    """
    c, mh, mw = protos.shape
    masks = np.matmul(masks_in, protos.reshape((c, -1))).reshape((-1, mh, mw)).transpose(1, 2, 0)  # HWN
    masks = np.ascontiguousarray(masks)
    masks = scale_mask(masks, im0_shape)  # re-scale mask from P3 shape to original input image shape
    masks = np.einsum("HWN -> NHW", masks)  # HWN -> NHW
    masks = crop_mask(masks, bboxes)
    return np.greater(masks, 0.5)

def scale_mask(masks: numpy.ndarray, 
               im0_shape: Tuple[int, int, int], 
               ratio_pad: Optional[Tuple[int, int]] = None):
    """
    Takes a mask, and resizes it to the original image size. (Borrowed from
    https://github.com/ultralytics/ultralytics/blob/465df3024f44fa97d4fad9986530d5a13cdabdca/ultralytics/utils/ops.py#L305)

    Args:
        masks (np.ndarray): resized and padded masks/images, [h, w, num]/[h, w, 3].
        im0_shape (tuple): the original image shape.
        ratio_pad (tuple): the ratio of the padding to the original image.

    Returns:
        masks (np.ndarray): The masks that are being returned.
    """
    im1_shape = masks.shape[:2]
    if ratio_pad is None:  # calculate from im0_shape
        gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # gain  = old / new
        pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # wh padding
    else:
        pad = ratio_pad[1]

    # Calculate tlbr of mask
    top, left = int(round(pad[1] - 0.1)), int(round(pad[0] - 0.1))  # y, x
    bottom, right = int(round(im1_shape[0] - pad[1] + 0.1)), int(round(im1_shape[1] - pad[0] + 0.1))
    if len(masks.shape) < 2:
        raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
    masks = masks[top:bottom, left:right]
    masks = cv2.resize(
        masks, (im0_shape[1], im0_shape[0]), interpolation=cv2.INTER_LINEAR
    )  # INTER_CUBIC would be better
    if len(masks.shape) == 2:
        masks = masks[:, :, None]
    return masks

    