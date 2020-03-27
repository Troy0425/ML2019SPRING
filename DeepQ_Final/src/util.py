"""Utility functions"""

import os
import sys
import numpy as np
import pydicom
from PIL import Image
sys.path.append("keras-retinanet")
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image


def dicom_to_jpg(in_file, out_file, out_size):
    """ Convert dicom file to jpg with specified size """
    ds = pydicom.read_file(in_file)
    size = (ds.Columns, ds.Rows)
    mode = 'L'
    im = Image.frombuffer(mode, size, ds.pixel_array,
                          "raw", mode, 0, 1).convert("L")
    im = im.resize((out_size, out_size), resample=Image.BICUBIC)
    im.save(out_file, quality=95)


def iou(box1, box2):
    """
    From Yicheng Chen's "Mean Average Precision Metric"
    https://www.kaggle.com/chenyc15/mean-average-precision-metric

    helper function to calculate IoU
    """
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2
    w1, h1 = x12-x11, y12-y11
    w2, h2 = x22-x21, y22-y21

    area1, area2 = w1 * h1, w2 * h2
    xi1, yi1, xi2, yi2 = max([x11, x21]), max(
        [y11, y21]), min([x12, x22]), min([y12, y22])

    if xi2 <= xi1 or yi2 <= yi1:
        return 0
    else:
        intersect = (xi2-xi1) * (yi2-yi1)
        union = area1 + area2 - intersect
        return intersect / union


def map_iou(boxes_true, boxes_pred, scores, thresholds=[0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]):
    """
    From Yicheng Chen's "Mean Average Precision Metric"
    https://www.kaggle.com/chenyc15/mean-average-precision-metric

    Mean average precision at differnet intersection over union (IoU) threshold

    input:
        boxes_true: Mx4 numpy array of ground true bounding boxes of one image. 
                    bbox format: (x1, y1, w, h)
        boxes_pred: Nx4 numpy array of predicted bounding boxes of one image. 
                    bbox format: (x1, y1, w, h)
        scores:     length N numpy array of scores associated with predicted bboxes
        thresholds: IoU shresholds to evaluate mean average precision on
    output: 
        map: mean average precision of the image
    """

    # According to the introduction, images with no ground truth bboxes will not be
    # included in the map score unless there is a false positive detection (?)

    # return None if both are empty, don't count the image in final evaluation (?)
    if len(boxes_true) == 0 and len(boxes_pred) == 0:
        return None

    assert boxes_true.shape[1] == 4 or boxes_pred.shape[1] == 4, "boxes should be 2D arrays with shape[1]=4"
    if len(boxes_pred):
        assert len(scores) == len(
            boxes_pred), "boxes_pred and scores should be same length"
        # sort boxes_pred by scores in decreasing order
        boxes_pred = boxes_pred[np.argsort(scores)[::-1], :]

    map_total = 0

    # loop over thresholds
    for t in thresholds:
        matched_bt = set()
        tp, fn = 0, 0
        for i, bt in enumerate(boxes_true):
            matched = False
            for j, bp in enumerate(boxes_pred):
                miou = iou(bt, bp)
                if miou >= t and not matched and j not in matched_bt:
                    matched = True
                    tp += 1  # bt is matched for the first time, count as TP
                    matched_bt.add(j)
            if not matched:
                fn += 1  # bt has no match, count as FN

        # FP is the bp that not matched to any bt
        fp = len(boxes_pred) - len(matched_bt)
        m = tp / (tp + fn + fp)
        map_total += m

    return map_total / len(thresholds)


def get_annotations(generator):
    """ Return list of annotations from generator """
    annotations = []
    for i in range(generator.size()):
        # load the annotations
        annotation = generator.load_annotations(i)[:, :4]
        annotations.append(annotation)
    return annotations


def get_scores(model, image, scale):
    """ Return calculated bounding boxes and scores for an image """
    # run network
    boxes, scores, labels = model.predict_on_batch(
        np.expand_dims(image, axis=0))

    # correct boxes for image scale
    boxes /= scale

    image_scores = scores[0]
    image_boxes = boxes[0]

    return (image_boxes, image_scores)


def get_view_from_dicom(dcmfile):
    """ Return ViewPosition dicom field from .dcm file """
    ds = pydicom.read_file(dcmfile)
    return ds.ViewPosition


def get_views_from_generator(generator, dcmdir):
    views = []
    for i in range(generator.size()):
        dcmfile = os.path.basename(generator.image_path(i))[:-4]+".dcm"
        dcmfpath = os.path.join(dcmdir, dcmfile)
        views.append(get_view_from_dicom(dcmfpath))
    return views


def get_detection_from_file(fpath, model, sz):
    image = read_image_bgr(fpath)
    image = preprocess_image(image)
    image, scale = resize_image(image, min_side=sz)
    return get_scores(model, image, scale)


def get_detections_from_generator(generator, model):
    detections = []

    for i in range(generator.size()):
        path = generator.image_path(i)
        raw_image = generator.load_image(i)
        image = generator.preprocess_image(raw_image.copy())
        image, scale = generator.resize_image(image)
        detections.append(get_scores(model, image, scale))
    return detections





