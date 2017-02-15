import numpy as np
from bnr_ml.objectdetect.utils import BoundingBox

import pdb

def average_precision(predictions, labels, cls, min_iou=0.5, return_pr_curve=False):
	'''
	Calculate average precision given predictions, labels, class.

	predictions - N x 5 matrix, first 4 are coordinates for bounding box,
		last is the confidence score.  These should be the restricted to the
		predictions for the specified class

	labels - list of dictionaries containing labels, this should be restricted to
		the relevant class.
	'''
	# order predictions in descending order of confidence
	idx = np.argsort(predictions[:,-1])[::-1]
	predictions = predictions[idx,:]
	num_labels = labels.__len__()
	was_used = np.zeros(num_labels, dtype=np.bool)
	tp, fp = np.zeros(predictions.shape[0]), np.zeros(predictions.shape[0])

	for i in range(predictions.shape[0]):
		pred = predictions[i]
		pred_box = BoundingBox(pred[0], pred[1], pred[0] + pred[2], pred[1] + pred[3])
		best_iou = -np.inf
		best_label = -1
		for j, label in enumerate(labels):
			gt_box = BoundingBox(label['x'], label['y'], label['x'] + label['w'], label['y'] + label['w'])
			if pred_box.iou(gt_box) > best_iou:
				best_iou = pred_box.iou(gt_box)
				best_label = j
				# was_used[j] = False

		if best_iou > min_iou:
			if not was_used[best_label]:
				tp[i] += 1.
			else:
				fp[i] += 1.
			was_used[best_label] = True
		else:
			fp[i] += 1.

	tp, fp = np.cumsum(tp), np.cumsum(fp)
	recall = tp / num_labels
	precision = tp / (tp + fp)

	# return precision, recall
	return _ap(precision, recall), precision, recall


def _ap(precision, recall):
	prec, rec = np.zeros(precision.size + 2), np.zeros(recall.size + 2)
	prec[1:-1], rec[1:-1] = precision, recall
	prec[0], prec[-1] = 0., 0.
	rec[0], rec[-1] = 0., 1.
	
	for i in range(prec.size - 2, -1, -1):
		prec[i] = max(prec[i], prec[i+1])

	index = np.asarray([i + 1 for i in range(rec.size - 1) if rec[i] != rec[i+1]])

	return ((rec[index] - rec[index - 1]) * prec[index]).sum()

def map(detector, annotations):
	pass
