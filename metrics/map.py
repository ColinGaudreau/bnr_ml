import numpy as np
from bnr_ml.objectdetect.utils import BoundingBox
from bnr_ml.utils.helpers import StreamPrinter
from skimage.io import imread

import pdb

def _ap_per_box(boxes, labels, min_iou=0.5, return_pr_curve=False):
	'''
	Calculate average precision given predictions, labels, class.

	labels - list of dictionaries containing labels, this should be restricted to
		the relevant class.
	'''
	boxes, scores = np.asarray(boxes), np.asarray([box.confidence for box in boxes])

	# order predictions in descending order of confidence
	idx = np.argsort(scores)[::-1]
	boxes = boxes[idx]
	num_labels = labels.__len__()
	was_used = np.zeros(num_labels, dtype=np.bool)
	tp, fp = np.zeros(boxes.size), np.zeros(boxes.size)

	for i in range(boxes.size):
		box = boxes[i]
		best_iou = 0.
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
	recall = tp / num_labels if num_labels > 0 else tp * 0
	precision = tp / (tp + fp)
	return _ap(precision, recall)

def _ap(precision, recall):
	prec, rec = np.zeros(precision.size + 2), np.zeros(recall.size + 2)
	prec[1:-1], rec[1:-1] = precision, recall
	prec[0], prec[-1] = 0., 0.
	rec[0], rec[-1] = 0., 1.
	for i in range(prec.size - 2, -1, -1):
		prec[i] = max(prec[i], prec[i+1])
	index = np.asarray([i + 1 for i in range(rec.size - 1) if np.abs(rec[i] - rec[i+1]) > 1e-5])
	return ((rec[index] - rec[index - 1]) * prec[index]).sum()

def map(detector, annotations, num_to_label, verbose=True, print_obj=StreamPrinter(open('/dev/stdout', 'w')), loc='', detector_args={}):
	aps = {}
	for _, label in num_to_label.iteritems():
		aps[label] = []
	detector_args.update({'num_to_label': num_to_label})
	
	if verbose:
		print_obj.println('Beginning mean average precision calculation...')
	for i, annotation in enumerate(annotations):
		labels = []
		# get all object types in image
		img_classes = np.unique([o['label'] for o in annotation['annotations']]).tolist()

		preds = detector.detect(imread(loc + annotation['image']), **detector_args)
		for cls in img_classes:
			labels = [label for label in annotation['annotations'] if label['label'] == cls]

			in_pred = False
			for box in preds:
				if box.cls == cls:
					in_pred = True
					break

			if in_pred:
				aps[cls].append(_ap_per_box([box for box in preds if box.cls == cls], labels))
			else:
				aps[cls].append(0.)

		if verbose:
			mean_ap = np.mean([np.mean(ap) for _, ap in aps.iteritems() if len(ap) > 0])
			print_obj.println('Annotation %d complete, mAP so far: %.3f' % (i, mean_ap))

	class_ap = {}
	for key, ap in aps.iteritems():
		class_ap[key] = np.mean(ap)
	return class_ap

def _precision_per_box(boxes, labels, min_iou=0.5, return_pr_curve=False):
	'''
	Calculate average precision given predictions, labels, class.

	labels - list of dictionaries containing labels, this should be restricted to
		the relevant class.
	'''
	boxes, scores = np.asarray(boxes), np.asarray([box.confidence for box in boxes])

	# order predictions in descending order of confidence
	idx = np.argsort(scores)[::-1]
	boxes = boxes[idx]
	num_labels = labels.__len__()
	was_used = np.zeros(num_labels, dtype=np.bool)
	tp, fp = 0., 0.
	classes = np.unique([box.cls for box in boxes])

	for i in range(boxes.size):
		box = boxes[i]
		best_iou = 0.
		best_label = -1
		for j, label in enumerate(labels):
			gt_box = BoundingBox(label['x'], label['y'], label['x'] + label['w'], label['y'] + label['w'])
			if label['label'] == box.cls and box.iou(gt_box) > best_iou:
				best_iou = box.iou(gt_box)
				best_label = j

		if best_iou > min_iou:
			if not was_used[best_label]:
				tp += 1.
			else:
				fp += 1.
			was_used[best_label] = True
		else:
			fp += 1.

	return tp / (tp + fp)

def precision(detector, annotations, num_to_label, verbose=True, print_obj=StreamPrinter(open('/dev/stdout', 'w')), detector_args={}):
	precisions = []
	detector_args.update({'num_to_label': num_to_label})
	
	if verbose:
		print_obj.println('Beginning precision calculation...')
	for i, annotation in enumerate(annotations):
		labels = []
		# get all object types in image
		img_classes = np.unique([o['label'] for o in annotation['annotations']]).tolist()

		boxes = detector.detect(imread(annotation['image']), **detector_args)

		if len(boxes) == 0:
			precisions.append(0.)
		else:
			precisions.append(_precision_per_box(boxes, annotation['annotations']))

		if verbose:
			mean_precision = np.mean(precisions)
			print_obj.println('Annotation %d complete, mean precision so far: %.3f' % (i, mean_precision))

	return np.mean(precisions)

def _f1_per_box(boxes, labels, min_iou=0.5, return_pr_curve=False):
	'''
	Calculate average precision given predictions, labels, class.

	labels - list of dictionaries containing labels, this should be restricted to
		the relevant class.
	'''
	boxes, scores = np.asarray(boxes), np.asarray([box.confidence for box in boxes])

	# order predictions in descending order of confidence
	idx = np.argsort(scores)[::-1]
	boxes = boxes[idx]
	num_labels = labels.__len__()
	was_used = np.zeros(num_labels, dtype=np.bool)
	tp, fp = 0., 0.
	classes = np.unique([box.cls for box in boxes])

	for i in range(boxes.size):
		box = boxes[i]
		best_iou = 0.
		best_label = -1
		for j, label in enumerate(labels):
			gt_box = BoundingBox(label['x'], label['y'], label['x'] + label['w'], label['y'] + label['w'])
			if label['label'] == box.cls and box.iou(gt_box) > best_iou:
				best_iou = box.iou(gt_box)
				best_label = j

		if best_iou > min_iou:
			if not was_used[best_label]:
				tp += 1.
			else:
				fp += 1.
			was_used[best_label] = True
		else:
			fp += 1.

	return 2*tp / (2*tp + fp + max(len(labels) - tp, 0))

def f1_score(detector, annotations, num_to_label, min_iou=0.5, verbose=True, print_obj=StreamPrinter(open('/dev/stdout', 'w')), loc='', detector_args={}):
	f1_scores = []
	detector_args.update({'num_to_label': num_to_label})
	
	if verbose:
		print_obj.println('Beginning f1 score calculation...')
	for i, annotation in enumerate(annotations):
		labels = []
		# get all object types in image
		img_classes = np.unique([o['label'] for o in annotation['annotations']]).tolist()

		boxes = detector.detect(imread(loc + annotation['image']), **detector_args)

		if len(boxes) == 0:
			f1_scores.append(0.)
		else:
			f1_scores.append(_f1_per_box(boxes, annotation['annotations'], min_iou=min_iou))

		if verbose:
			print_obj.println('Annotation %d complete, mean f1 score so far: %.3f' % (i, np.mean(f1_scores)))

	return np.mean(f1_scores)


		

	



