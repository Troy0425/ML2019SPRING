#!/usr/bin/env python3
"""
Script for predicting bounding boxes for the RSNA pneumonia detection challenge
by Phillip Cheng, MD MS
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
# import utility functions

# This is a modified version of keras-retinanet 0.4.1
# which includes a score metric to estimate the RSNA score
# at the threshold giving the maximum Youden index.
#sys.path.append("keras-retinanet")
from keras_retinanet import models

def wt_av(x, xw, y, yw):
	""" Calculate a weighted average """
	return (x*xw+y*yw)/(xw+yw)
def nms(boxes, scores, overlapThresh):
	"""
	adapted from non-maximum suppression by Adrian Rosebrock
	https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
	"""

	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return np.array([]).reshape(0, 4), np.array([])
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
	pick = []
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]

	# compute the area of the bounding boxes
	area = (x2 - x1 + 1) * (y2 - y1 + 1)

	# sort the bounding boxes by scores in ascending order
	idxs = np.argsort(scores, axis=0)
	idxs = idxs.reshape((len(idxs)))

	# keep looping while indexes still remain in the indexes list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]

		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
											   np.where(overlap > overlapThresh)[0])))
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick], scores[pick]

def averages(boxes, scores, overlapThresh, solo_min=0):
	""" Like non-max-suppression, but take weighted averages of overlapping bounding boxes """
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return np.array([]).reshape(0, 4), np.array([])
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	pick = []
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]

	# compute the area of the bounding boxes
	area = (x2 - x1 + 1) * (y2 - y1 + 1)

	# sort the bounding boxes by scores in ascending order
	idxs = np.argsort(scores, axis=0)
	idxs = idxs.reshape((len(idxs)))

	# keep looping while indexes still remain in the indexes list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]

		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]

		overlap_idx = np.where(overlap > overlapThresh)[0].tolist()[::-1]

		if len(overlap_idx) == 0:
			if scores[i] >= solo_min:
				pick.append(i)
		else:
			pick.append(i)
			for j in overlap_idx:
				boxes[i, :] = wt_av(boxes[i, :], scores[i],
									boxes[idxs[j], :], scores[idxs[j]])
				scores[i] = scores[i]+scores[idxs[j]]

		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
											   np.where(overlap > overlapThresh)[0])))

	# return only the bounding boxes that were picked using the
	# integer data type
	if len(pick) > 0:
		return boxes[pick], scores[pick]
	else:
		return np.array([]).reshape(0, 4), np.array([])


def intersects(boxes, scores, overlapThresh, solo_min=0, shrink=0):
	""" Like weighted averages, but take intersections of overlapping bounding boxes """

	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return np.array([]).reshape(0, 4), np.array([])
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	pick = []
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]

	# compute the area of the bounding boxes
	area = (x2 - x1 + 1) * (y2 - y1 + 1)

	# sort the bounding boxes by scores in ascending order
	idxs = np.argsort(scores, axis=0)
	idxs = idxs.reshape((len(idxs)))

	# keep looping while indexes still remain in the indexes list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]

		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]

		overlap_idx = np.where(overlap > overlapThresh)[0].tolist()[::-1]

		if len(overlap_idx) == 0:
			if scores[i] >= solo_min:
				pick.append(i)

				shrink_factor = shrink/2
				(bx1, by1, bx2, by2) = boxes[i, :]
				diffx = bx2-bx1
				diffy = by2-by1
				boxes[i, 0] += shrink_factor*diffx
				boxes[i, 1] -= shrink_factor*diffx
				boxes[i, 2] += shrink_factor*diffy
				boxes[i, 3] -= shrink_factor*diffy
		else:
			pick.append(i)
			for j in overlap_idx:
				boxes[i, :] = (xx1[j], yy1[j], xx2[j], yy2[j])
				scores[i] = scores[i]+scores[idxs[j]]

		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
											   np.where(overlap > overlapThresh)[0])))

	# return only the bounding boxes that were picked using the
	# integer data type
	if len(pick) > 0:
		return boxes[pick], scores[pick]
	else:
		return np.array([]).reshape(0, 4), np.array([])

def shrink(bb, shrink_factor):
	""" Shrinks bounding boxes by a factor in each dimension """
	if len(bb) > 0:
		x1 = bb[:, 0]
		y1 = bb[:, 1]
		x2 = bb[:, 2]
		y2 = bb[:, 3]
		diffx = x2-x1
		diffy = y2-y1
		shrink_factor /= 2
		x1 += shrink_factor*diffx
		x2 -= shrink_factor*diffx
		y1 += shrink_factor*diffy
		y2 -= shrink_factor*diffy



model = models.load_model("./model1.h5", backbone_name='resnet50')

test_jpg_dir = "../data/test/"
submission_dir = "./"

sz = int(sys.argv[1])

# threshold for non-max-suppresion for each model
nms_threshold = 0

# shrink bounding box dimensions by this factor, improves test set performance
shrink_factor = 0.17

# threshold for judging overlap of bounding boxes between different networks (for weighted average)
wt_overlap = 0

# threshold for including boxes from model 1
score_threshold1 =  0.04   #0.04

solo_min = 0.15

test_ids = []
test_outputs = []

start = time.time()

for i, fname in enumerate(os.listdir(test_jpg_dir)):
	print(f"Predicting boxes for image # {i+1}\r", end="")
	fpath = os.path.join(test_jpg_dir, fname)
	fid = fname[:-4]

	image = read_image_bgr(fpath)
	image = preprocess_image(image)
	image, scale = resize_image(image, min_side=sz)
	boxes, scores,labels =model.predict_on_batch(np.expand_dims(image, axis=0))
	boxes /= scale
	scores = scores[0]
	boxes_pred = boxes[0]

	indices = np.where(scores > score_threshold1)[0]
	scores = scores[indices]
	boxes_pred = boxes_pred[indices]
	boxes_pred, scores = nms(boxes_pred, scores, nms_threshold)

	boxes_pred, scores = averages(boxes_pred, scores, wt_overlap, solo_min)
	shrink(boxes_pred, shrink_factor)

	output = ''
	for j, bb in enumerate(boxes_pred):
		x1 = int(bb[0])
		y1 = int(bb[1])
		w = int(bb[2]-x1+1)
		h = int(bb[3]-y1+1)
		output += "0.0 %d %d %d %d "%(x1,y1,w,h)
	test_ids.append(fid)
	test_outputs.append(output)
print()
end = time.time()
# print execution time
print(f"Elapsed time = {end-start:.3f} seconds")

test_df = pd.DataFrame({'patientId': test_ids, 'PredictionString': test_outputs},
					   columns=['patientId', 'PredictionString'])
if not os.path.exists(submission_dir):
	os.mkdir(submission_dir)
test_df.to_csv(os.path.join(submission_dir, "ans.csv"), index = False)
