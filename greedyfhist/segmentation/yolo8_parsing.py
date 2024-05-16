from typing import Tuple, List

import cv2
import numpy
import numpy as np

# import skimage

# def crop_mask(masks: numpy.array, boxes: numpy.array) -> numpy.array:
#     """
#     It takes a mask and a bounding box, and returns a mask that is cropped to the bounding box.

#     Args:
#         masks (torch.Tensor): [n, h, w] tensor of masks
#         boxes (torch.Tensor): [n, 4] tensor of bbox coordinates in relative point form

#     Returns:
#         (torch.Tensor): The masks are being cropped to the bounding box.
#     """
#     n, h, w = masks.shape
#     # x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(n,1,1)
#     x1, y1, x2, y2 = np.split(boxes, 4, 1)
    
#     #r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,1,w)
#     #c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(1,h,1)
#     r = np.arange(w, dtype=x1.dtype)[None, None, :]  # rows shape(1,1,w)
#     c = np.arange(h, dtype=x1.dtype)[None, :, None]  # cols shape(1,h,1)

#     return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


# def compute_iou(box: numpy.array, boxes: numpy.array) -> float:
#     # Compute xmin, ymin, xmax, ymax for both boxes
#     xmin = np.maximum(box[0], boxes[:, 0])
#     ymin = np.maximum(box[1], boxes[:, 1])
#     xmax = np.minimum(box[2], boxes[:, 2])
#     ymax = np.minimum(box[3], boxes[:, 3])

#     # Compute intersection area
#     intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

#     # Compute union area
#     box_area = (box[2] - box[0]) * (box[3] - box[1])
#     boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
#     union_area = box_area + boxes_area - intersection_area

#     # Compute IoU
#     iou = intersection_area / union_area
#     return iou


# def nms(boxes: numpy.array, scores: numpy.array, iou_threshold: float) -> List[float]:
#     # TODO: Replace with opencvs nms function?!
#     # Sort by score
#     sorted_indices = np.argsort(scores)[::-1]

#     keep_boxes = []
#     while sorted_indices.size > 0:
#         # Pick the last box
#         box_id = sorted_indices[0]
#         keep_boxes.append(box_id)

#         # Compute IoU of the picked box with the rest
#         ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

#         # Remove boxes with IoU over the threshold
#         keep_indices = np.where(ious < iou_threshold)[0]

#         # print(keep_indices.shape, sorted_indices.shape)
#         sorted_indices = sorted_indices[keep_indices + 1]

#     return keep_boxes


# def non_max_suppression(
#         prediction: numpy.array,
#         conf_thres: float = 0.25,
#         multi_label: bool = False,
#         rotated: bool = False,
#         max_nms: int = 30000,
#         max_wh: int = 7680,
#         max_det: int= 300,
#         agnostic: bool = False,
#         nc: int = 1,
#         iou_thres: float =.7) -> List[numpy.array]:

#     bs = prediction.shape[0]  # batch size
#     nc = nc or (prediction.shape[1] - 4)  # number of classes
#     nm = prediction.shape[1] - nc - 4
#     mi = 4 + nc  # mask start index
#     xc = prediction[:, 4:mi].max(1) > conf_thres  # candidates

#     # Settings
#     # min_wh = 2  # (pixels) minimum box width and height
#     multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

#     prediction = prediction.transpose([0, -1, -2])  # shape(1,84,6300) to shape(1,6300,84)
#     if not rotated:
#         prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy

#     # output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
#     output = [np.zeros((0, 6 + nm))] * bs
#     for xi, x in enumerate(prediction):  # image index, image inference
#         # Apply constraints
#         x = x[xc[xi]]  # confidence

#         # Cat apriori labels if autolabelling
#         # If none remain process next image
#         if not x.shape[0]:
#             continue

#         # Detections matrix nx6 (xyxy, conf, cls)
#         box = x[:, :4]
#         cls = x[:, 4:(4 + nc)]
#         mask = x[:, (4 + nc):(4 + nc + nm)]

#         conf = cls.max(1, keepdims=True)
#         j = np.argmax(cls, axis=1)
#         j = j.reshape((j.shape[0],1))
#         x = np.hstack((box, conf, j, mask))[np.squeeze(conf) > conf_thres]

#         # Check shape
#         n = x.shape[0]  # number of boxes
#         if not n:  # no boxes
#             continue
#         if n > max_nms:  # excess boxes
#             x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

#         # Batched NMS
#         c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
#         scores = x[:, 4]  # scores
#         boxes = x[:, :4] + c  # boxes (offset by class)
#         # break
#         i = nms(boxes, scores, iou_thres)  # NMS
#         i = i[:max_det]  # limit detections

#         output[xi] = x[i]
#     return output


# def sigmoid(x: numpy.array) -> numpy.array:
#     return 1 / (1 + np.exp(-x))


# def process_mask(protos: numpy.array, 
#                  masks_in: numpy.array, 
#                  bboxes: numpy.array, 
#                  shape: Tuple[int, int], 
#                  upsample: bool = True) -> numpy.array:
#     c, mh, mw = protos.shape
#     ih, iw = shape
#     masks = sigmoid(masks_in @ protos.reshape((c, -1))).reshape((-1, mh, mw))
#     downsampled_bboxes = bboxes.copy()
#     downsampled_bboxes[:, 0] *= mw / iw
#     downsampled_bboxes[:, 2] *= mw / iw
#     downsampled_bboxes[:, 3] *= mh / ih
#     downsampled_bboxes[:, 1] *= mh / ih
#     masks = crop_mask(masks, downsampled_bboxes)  # CHW
#     if upsample:
#         masks = skimage.transform.resize(masks, [masks.shape[0], shape[0], shape[1]], anti_aliasing=True)
#     masks[masks < 0.5] = 0
#     masks[masks > 0] = 1
#     return masks

# def xywh2xyxy(x: numpy.array) -> numpy.array:
#     """
#     Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
#     top-left corner and (x2, y2) is the bottom-right corner.

#     Args:
#         x (np.ndarray): The input bounding box coordinates in (x, y, width, height) format.

#     Returns:
#         y (np.ndarray): The bounding box coordinates in (x1, y1, x2, y2) format.
#     """
#     assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
#     y = np.empty_like(x)  # faster than clone/copy
#     dw = x[..., 2] / 2  # half-width
#     dh = x[..., 3] / 2  # half-height
#     y[..., 0] = x[..., 0] - dw  # top left x
#     y[..., 1] = x[..., 1] - dh  # top left y
#     y[..., 2] = x[..., 0] + dw  # bottom right x
#     y[..., 3] = x[..., 1] + dh  # bottom right y
#     return y


# def postprocess_segmentation(outputs: Tuple[numpy.array, numpy.array], 
#                              image_shape: Tuple[int, int], 
#                              upsample: bool = True) -> numpy.array:
#     """
#     Reimplementation of the postprocessing routine for segmentation from Ultralytics. Only implements
#     relevant parts, i.e. some functions are missing.    
#     """
#     p = non_max_suppression(outputs[0].copy())
#     proto = outputs[1].copy()
#     masks = []
#     for i, pred in enumerate(p):
#         protos = proto[i]
#         masks_in = pred[:, 6:]
#         bboxes = pred[:, :4]
#         mask = process_mask(protos, masks_in, bboxes, image_shape, upsample)
#         masks.append(mask)
#     return masks


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

def masks2segments(masks):
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

def crop_mask(masks, boxes):
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

def process_mask(protos, masks_in, bboxes, im0_shape):
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

def scale_mask(masks, im0_shape, ratio_pad=None):
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

    