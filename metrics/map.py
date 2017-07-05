import numpy as np
from bnr_ml.objectdetect.utils import BoundingBox
from bnr_ml.utils.helpers import StreamPrinter
from skimage.io import imread

import pdb

def _confusion_per_box(boxes, labels, min_iou=0.5, return_pr_curve=False):
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

def confusion(detector, annotations, num_to_label, verbose=True, print_obj=StreamPrinter(open('/dev/stdout', 'w')), loc='', detector_args={}):
	confusion_data = {}

	for _, label in num_to_label.iteritems():
		confusion_data[label] = {'scores': [], 'tp': [], 'fp': [], 'n_labels': []}
	detector_args.update({'num_to_label': num_to_label})
	
	if verbose:
		print_obj.println('Doing confusion matrix calculation...')
	for i, annotation in enumerate(annotations):
		labels = []
		# get all object types in image
		img_classes = np.unique([o['label'] for o in annotation['annotations']]).tolist()

		boxes = detector.detect(imread(loc + annotation['image']), **detector_args)
		for cls in img_classes:
			labels = [label for label in annotation['annotations'] if label['label'] == cls]

			in_pred = False
			for box in boxes:
				if box.cls == cls:
					in_pred = True
					break

			if in_pred:
				scores, tp, fp, n_labels = _confusion_per_box([box for box in boxes if box.cls == cls], labels)
				confusion_data[cls]['scores'].extend(scores.tolist())
				confusion_data[cls]['tp'].extend(tp.tolist())
				confusion_data[cls]['fp'].extend(fp.tolist())
				confusion_data[cls]['n_labels'].append(n_labels)

		if verbose:
			print_obj.flush()
			print_obj.write('\rAnnotation %d/%d' % (i+1,len(annotations)))

	confusion = {}
	for key, data in confusion_data.iteritems():
		scores, tp, fp, n_labels = np.asarray(data['scores']), np.asarray(data['tp']), np.asarray(data['fp']), np.sum(data['n_labels'])
		idx = np.argsort(scores)[::-1]
		tp, fp = tp[idx], fp[idx]
		tp, fp = np.cumsum(tp), np.cumsum(fp)
		fn = np.maximum(0., n_labels - tp)
		confusion[key] = {'tp': tp, 'fp': fp, 'fn': fn}

	return confusion

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
	if verbose:
		print_obj.println('Beginning mean average precision calculation...')

	conf_mat = confusion(detector, annotations, num_to_label, verbose=verbose, print_obj=print_obj, loc=loc, detector_args=detector_args)

	class_ap = {}
	for key, conf in conf_mat.iteritems():
		tp, fp, fn = conf['tp'], conf['fp'], conf['fn']

		recall = np.nan_to_num(tp / (tp + fn), 0.)
		precision = tp / (tp + fp)

		a, precision, recall = _ap(precision, recall)

		class_ap[key] = {'ap': a, 'precision': precision, 'recall': recall}
	return class_ap

# def _precision_per_box(boxes, labels, min_iou=0.5, return_pr_curve=False):
# 	'''
# 	Calculate average precision given predictions, labels, class.

# 	labels - list of dictionaries containing labels, this should be restricted to
# 		the relevant class.
# 	'''
# 	boxes, scores = np.asarray(boxes), np.asarray([box.confidence for box in boxes])

# 	# order predictions in descending order of confidence
# 	idx = np.argsort(scores)[::-1]
# 	boxes = boxes[idx]
# 	num_labels = labels.__len__()
# 	was_used = np.zeros(num_labels, dtype=np.bool)
# 	tp, fp = 0., 0.
# 	classes = np.unique([box.cls for box in boxes])

# 	for i in range(boxes.size):
# 		box = boxes[i]
# 		best_iou = 0.
# 		best_label = -1
# 		for j, label in enumerate(labels):
# 			gt_box = BoundingBox(label['x'], label['y'], label['x'] + label['w'], label['y'] + label['w'])
# 			if label['label'] == box.cls and box.iou(gt_box) > best_iou:
# 				best_iou = box.iou(gt_box)
# 				best_label = j

# 		if best_iou > min_iou:
# 			if not was_used[best_label]:
# 				tp += 1.
# 			else:
# 				fp += 1.
# 			was_used[best_label] = True
# 		else:
# 			fp += 1.

# 	return tp / (tp + fp)

# def precision(detector, annotations, num_to_label, verbose=True, print_obj=StreamPrinter(open('/dev/stdout', 'w')), detector_args={}):
# 	precisions = []
# 	detector_args.update({'num_to_label': num_to_label})
	
# 	if verbose:
# 		print_obj.println('Beginning precision calculation...')
# 	for i, annotation in enumerate(annotations):
# 		labels = []
# 		# get all object types in image
# 		img_classes = np.unique([o['label'] for o in annotation['annotations']]).tolist()

# 		boxes = detector.detect(imread(annotation['image']), **detector_args)

# 		if len(boxes) == 0:
# 			precisions.append(0.)
# 		else:
# 			precisions.append(_precision_per_box(boxes, annotation['annotations']))

# 		if verbose:
# 			mean_precision = np.mean(precisions)
# 			print_obj.println('Annotation %d complete, mean precision so far: %.3f' % (i, mean_precision))

# 	return np.mean(precisions)

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
	if verbose:
		print_obj.println('Beginning f1 score calculation...')

	conf_mat = confusion(detector, annotations, num_to_label, verbose=verbose, print_obj=print_obj, loc=loc, detector_args=detector_args)

	f1_scores = {}
	for cls, conf in conf_mat.iteritems():
		tp, fp, fn = conf['tp'], conf['fp'], conf['fn']
		if tp.size == 0:
			f1_scores[cls] = 0.
			continue
		tp, fp, fn = tp[-1], fp[-1], fn[-1]

		f1 = (1 + beta**2) * tp / ((1 + beta**2) * tp + fp + (beta**2) * fn)

		f1_scores[cls] = f1

	return f1_scores

def confusion(detector, annotations, num_to_label, verbose=True, print_obj=StreamPrinter(open('/dev/stdout', 'w')), loc='', detector_args={}):
	confusion_data = {}

	for _, label in num_to_label.iteritems():
		confusion_data[label] = {'scores': [], 'tp': [], 'fp': [], 'n_labels': []}
	detector_args.update({'num_to_label': num_to_label})
	
	if verbose:
		print_obj.println('Doing confusion matrix calculation...')
	for i, annotation in enumerate(annotations):
		labels = []
		# get all object types in image
		img_classes = np.unique([o['label'] for o in annotation['annotations']]).tolist()

		boxes = detector.detect(imread(loc + annotation['image']), **detector_args)
		for cls in img_classes:
			labels = [label for label in annotation['annotations'] if label['label'] == cls]

			in_pred = False
			for box in boxes:
				if box.cls == cls:
					in_pred = True
					break

			if in_pred:
				scores, tp, fp, n_labels = _confusion_per_box([box for box in boxes if box.cls == cls], labels)
				confusion_data[cls]['scores'].extend(scores.tolist())
				confusion_data[cls]['tp'].extend(tp.tolist())
				confusion_data[cls]['fp'].extend(fp.tolist())
				confusion_data[cls]['n_labels'].append(n_labels)

		if verbose:
			print_obj.flush()
			print_obj.write('\rAnnotation %d/%d' % (i+1,len(annotations)))

	confusion = {}
	for key, data in confusion_data.iteritems():
		scores, tp, fp, n_labels = np.asarray(data['scores']), np.asarray(data['tp']), np.asarray(data['fp']), np.sum(data['n_labels'])
		idx = np.argsort(scores)[::-1]
		tp, fp = tp[idx], fp[idx]
		tp, fp = np.cumsum(tp), np.cumsum(fp)
		fn = np.maximum(0., n_labels - tp)
		confusion[key] = {'tp': tp, 'fp': fp, 'fn': fn}

	return confusion

def _confusion_per_box(boxes, labels, min_iou=0.5, return_pr_curve=False):
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

def detector_analysis(detector, annotations, class_list, beta=1., min_iou=0.5, verbose=True, print_obj=StreamPrinter(open('/dev/stdout', 'w')), loc='', detector_args{}):
	if verbose:
		print_obj.println('Beginning detector analysis...')

	# initialize dict with all information to be returned
	info_dict = {
		'mAP': None, # final mAP of detector
		'f1': None, # final f1 score of detector
		'precision': None,
		'recall': None,
		'tp': [],
		'fp': [],
		'scores': [],
		'classes': [],
		'mAP-per-detection': [],
		'per-class': {},
		'fn-info': {'class': [], 'size': []},
		'fp-info': {'matched': {'class': [], 'size': [], 'scores': }, 'unmatched': {'class': [], 'scores': []}},
		'tp-info': {'class': [], 'size': [], 'scores': []}
	}
	n_labels = 0
	n_labels_class = {}
	for cls in class_list:
		n_labels_class[cls] = 0

	for i, annotation in enumerate(annotations):
		# perform detection in image
		boxes = np.asarray(detector.detect(imread(loc + annotation['image']), **detector_args))

		# get true labels
		labels = np.asarray(annotation['annotations'])

		# get scores from predictions and sort them in descending order
		scores = np.asarray([box.confidence for box in boxes])
		idx = np.argsort(scores)[::-1]

		was_used = np.zeros(labels.size, dtype=np.bool)

		tp, fp = np.zeros(boxes.size), np.zeros(boxes.size)

		# keep track of number of objects, also for each class
		n_labels += labels.size
		for cls in class_list:
			n_labels_class[cls] += np.sum([1 for obj in labels if obj['label'] == cls])

		for j, box in enumerate(boxes):
			best_iou = 0.
			best_label = -1
			best_gt = None
			info_dict['classes'].append(box.cls)

			# try to match ground truth box to prediction
			for k, label in enumerate(labels):
				gt_box = BoundingBox(label['x'], label['y'], label['x'] + label['w'], label['y'] + label['w'], cls=label['label'])
				if box.iou(gt_box) > best_iou and box.cls == gt_box.cls:
					best_iou = box.iou(gt_box)
					best_label = k
					best_gt = gt_box

			if best_iou > min_iou:
				if not was_used[best_label]:
					tp[j] += 1.
					info_dict['tp-info']['class'].append(best_gt.cls)
					info_dict['tp-info']['size'].append(best_gt.size)
					info_dict['tp-info']['scores'].append(box.confidence)
				else:
					fp[j] += 1.
					info_dict['fp-info']['matched']['class'].append(best_gt.cls) # store matched box class
					info_dict['fp-info']['matched']['size'].append(best_gt.size) # store matched box size
					info_dict['fp-info']['matched']['size'].append(box.confidence)
				was_used[best_label] = True
			else:
				fp[j] += 1.
				info_dict['fp-info']['unmatched']['class'].append(box.cls)
				info_dict['fp-info']['unmatched']['scores'].append(box.confidence)

		# look at false negative classes and sizes
		for j, used in enumerate(was_used):
			if not used:
				label = labels[j]
				gt_box = BoundingBox(label['x'], label['y'], label['x'] + label['w'], label['y'] + label['w'], cls=label['label'])
				info_dict['fn-info']['class'].append(gt_box.cls)
				info_dict['fn-info']['size'].append(gt_box.size)

		tp_image, fp_image = np.cumsum(tp), np.cumsum(fp)
		prec = tp_image / labels.size if labels.size > 0 else tp * 0
		rec = tp_image / (tp_image + fp_image)

		m_ap, _, _2 = _ap(prec, rec)
		# get quality of detection per image
		info_dict['mAP-per-detection'].append(m_ap)

		# store tp, fp, scores to calculate total mAP at the end
		info_dict['tp'].extend(tp.tolist())
		info_dict['fp'].extend(fp.tolist())
		info_dict['scores'].extend(scores.tolist())

		if verbose:
			print_obj.flush()
			print_obj.write('\rAnnotation %d/%d' % (i+1,len(annotations)))

	tp, fp, scores, classes = np.asarray(info_dict['tp']), np.asarray(info_dict['fp']), np.asarray(info_dict['scores']), np.asarray(info_dict['classes'])
	idx = np.argsort(scores)[::-1]
	tp, fp, scores, classes = tp[idx], fp[idx], scores[idx], classes[idx]

	# find mAP for each class
	for cls in class_list:
		idx_cls = classes == cls
		tp_cls, fp_cls = tp[idx_cls], fp[idx_cls]
		tp_cls, fp_cls = np.cumsum(tp_cls), np.cumsum(fp_cls)
		prec = tp_cls / n_labels_class[cls]
		rec = tp_cls / (tp_cls + fp_cls)
		m_ap, prec, rec = _ap(prec, rec)
		info_dict['per-class'][cls] = {'mAP': m_ap, 'precision': prec, 'recall': rec}

	# calculate mAP
	tp, fp = np.cumsum(tp), np.cumsum(fp)
	fn = n_labels - tp
	prec = tp / n_labels
	rec = tp / (tp + fp)
	m_ap, prec, rec = _ap(prec, rec)

	# store final calculations
	info_dict['precision'] = prec
	info_dict['recall'] = rec
	info_dict['mAP'] = m_ap
	info_dict['f1'] = (1 + beta**2) * tp / ((1 + beta**2) * tp + (beta**2) * fn + fp)
	info_dict['tp'] = tp
	info_dict['fp'] = fp
	info_dict['scores'] = scores

	return info_dict



