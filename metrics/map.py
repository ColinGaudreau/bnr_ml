import numpy as np
from bnr_ml.objectdetect.utils import BoundingBox
from bnr_ml.helpers.utils import StreamPrinter
from skimage.io import imread

import pdb

def average_precision(boxes, scores, labels, min_iou=0.5, return_pr_curve=False):
	'''
	Calculate average precision given predictions, labels, class.

	predictions - N x 5 matrix, first 4 are coordinates for bounding box,
		last is the confidence score.  These should be the restricted to the
		predictions for the specified class

	labels - list of dictionaries containing labels, this should be restricted to
		the relevant class.
	'''
	assert(boxes.__len__() == scores.__len__())

	boxes, scores = np.asarray(boxes), np.asarray(scores)

	# order predictions in descending order of confidence
	idx = np.argsort(scores)[::-1]
	boxes = boxes[idx]
	num_labels = labels.__len__()
	was_used = np.zeros(num_labels, dtype=np.bool)
	tp, fp = np.zeros(boxes.size), np.zeros(boxes.size)

	for i in range(boxes.size):
		box = boxes[i]
		best_iou = -np.inf
		best_label = -1
		for j, label in enumerate(labels):
			gt_box = BoundingBox(label['x'], label['y'], label['x'] + label['w'], label['y'] + label['w'])
			if box.iou(gt_box) > best_iou:
				best_iou = box.iou(gt_box)
				best_label = j

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

	return _ap(precision, recall)

def _ap(precision, recall):
	prec, rec = np.zeros(precision.size + 2), np.zeros(recall.size + 2)
	prec[1:-1], rec[1:-1] = precision, recall
	prec[0], prec[-1] = 0., 0.
	rec[0], rec[-1] = 0., 1.
	
	for i in range(prec.size - 2, -1, -1):
		prec[i] = max(prec[i], prec[i+1])

	index = np.asarray([i + 1 for i in range(rec.size - 1) if rec[i] != rec[i+1]])
	return ((rec[index] - rec[index - 1]) * prec[index]).sum()

def map(detector, annotations, num_to_label, verbose=True, print_obj=StreamPrinter(open('/dev/stdout', 'w')), detector_args={}):
	aps = []
	detector_args.update(num_to_label)
	if verbose:
		print_obj.println('Beginning mean average precision calculation...')
	for i, annotation in enumerate(annotations):
		labels = []
		predictions = detector(imread(annotation['image']), **detector_args)
		for key, cls in num_to_label.iteritems():
			labels = [label for label in annotation['annotations'] if label['label'] == cls]
			aps.append(average_precision(predictions[cls]['boxes'], predictions[cls]['scores'], labels))
		if verbose:
			print_obj.println('Annotation %d complete, mAP so far: %3f' % (i, np.mean(aps)))

	return np.mean(aps)


		

	



