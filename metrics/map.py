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
	
	# tp, fp = np.cumsum(tp), np.cumsum(fp)
	# recall = tp / num_labels if num_labels > 0 else tp * 0
	# precision = tp / (tp + fp)
	return scores[idx], tp, fp, num_labels

def _ap(precision, recall):
	prec, rec = np.zeros(precision.size + 2), np.zeros(recall.size + 2)
	prec[1:-1], rec[1:-1] = precision, recall
	prec[0], prec[-1] = 0., 0.
	rec[0], rec[-1] = 0., 1.
	for i in range(prec.size - 2, -1, -1):
		prec[i] = max(prec[i], prec[i+1])
	index = np.asarray([i + 1 for i in range(rec.size - 1) if np.abs(rec[i] - rec[i+1]) > 1e-5])
	return ((rec[index] - rec[index - 1]) * prec[index]).sum(), prec, rec

def map(detector, annotations, num_to_label, verbose=True, print_obj=StreamPrinter(open('/dev/stdout', 'w')), loc='', detector_args={}):
	aps = {}
	for _, label in num_to_label.iteritems():
		aps[label] = {'scores': [], 'tp': [], 'fp': [], 'n_labels': []}
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
				scores, tp, fp, n_labels = _ap_per_box([box for box in preds if box.cls == cls], labels)
				aps[cls]['scores'].extend(scores.tolist())
				aps[cls]['tp'].extend(tp.tolist())
				aps[cls]['fp'].extend(fp.tolist())
				aps[cls]['n_labels'].append(n_labels)
			# else:
				# aps[cls]['ap'].append(0.)

		if verbose:
			# mean_ap = np.mean([np.mean(ap['ap']) for _, ap in aps.iteritems() if len(ap['ap']) > 0])
			print_obj.flush()
			print_obj.write('\rAnnotation %d/%d' % (i+1,len(annotations)))
			# print_obj.write('\rAnnotation %d complete, mAP so far: %.3f' % (i, mean_ap))

	if verbose:
		print_obj.println('')

	class_ap = {}
	for key, ap in aps.iteritems():
		scores, tp, fp, n_labels = np.asarray(ap['scores']), np.asarray(ap['tp']), np.asarray(ap['fp']), np.sum(ap['n_labels'])
		idx = np.argsort(scores)[::-1]
		pdb.set_trace()
		tp, fp = tp[idx], fp[idx]
		tp, fp = np.cumsum(tp), np.cumsum(fp)
		recall = tp / n_labels if n_labels > 0 else np.zeros_like(tp)
		precision = tp / (tp + fp)
		a, precision, recall = _ap(precision, recall)
		class_ap[key] = {'ap': a, 'precision': precision, 'recall': recall}
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

def _f1_per_box(boxes, labels, beta=1., min_iou=0.5, return_pr_curve=False):
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

	return (1 + beta**2)*tp / ((1+beta**2)*tp + fp + (beta**2)*max(len(labels) - tp, 0))

def f1_score(detector, annotations, num_to_label, beta=1., min_iou=0.5, verbose=True, print_obj=StreamPrinter(open('/dev/stdout', 'w')), loc='', detector_args={}):
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
			f1_scores.append(_f1_per_box(boxes, annotation['annotations'], beta=beta, min_iou=min_iou))

		if verbose:
			print_obj.flush()
			print_obj.write('\rAnnotation %d complete, mean f1 score so far: %.3f' % (i, np.mean(f1_scores)))

	return np.mean(f1_scores)


		

	



