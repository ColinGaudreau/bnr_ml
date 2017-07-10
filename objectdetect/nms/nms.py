from bnr_ml.objectdetect.utils import BoundingBox
import copy
import numpy as np
import ap_nms
import ap_nms_gpu

import pdb

METHOD_VIOLA_JONES = 'viola-jones'
METHOD_GREEDY = 'greedy'
METHOD_AP = 'ap'
METHOD_AP_GPU = 'ap-gpu'

def nms(boxes, *args, **kwargs):
	'''
	Takes list of BoundingBox objects and does non maximal suppression.
	'''
	# nms for each class
	n_apply = 1
	if 'n_apply' in kwargs:
		n_apply = kwargs['n_apply']

	classes = list(set([box.cls for box in boxes]))
	boxes = np.asarray(copy.deepcopy(boxes))

	method = METHOD_VIOLA_JONES
	if 'method' in kwargs:
		method = kwargs['method']

	uses_iou = False
	
	if method.lower() == METHOD_VIOLA_JONES:
		detect_fn = lambda boxes, *args, **kwargs: _viola_jones(boxes, *args, **kwargs)
	elif method.lower() == METHOD_GREEDY:
		detect_fn = lambda boxes, *args, **kwargs: _greedy(boxes, *args, **kwargs)
	elif method.lower() == METHOD_AP:
		assert('iou' in kwargs)
		detect_fn = lambda boxes, *args, **kwargs: ap_nms.affinity_propagation(boxes, *args, **kwargs)
		uses_iou = True
	elif method.lower() == METHOD_AP_GPU:
		assert('iou' in kwargs)
		detect_fn = lambda boxes, *args, **kwargs: ap_nms_gpu.affinity_propagation(boxes, *args, **kwargs)
		uses_iou = True
	else:
		raise Exception('Method "{}" not valid.'.format(method))

	idx = np.arange(boxes.size)
	for _ in range(n_apply):
		objs = []
		for cls in classes:
			idx_cls = idx[np.asarray([box.cls == cls for box in boxes])]
			if uses_iou:
				idx_row, idx_col = np.meshgrid(idx_cls, idx_cls)
				affinity = kwargs['iou'][idx_row, idx_col] - 1.
				diag = np.arange(affinity.shape[0])
				affinity[diag,diag] = np.asarray([b.confidence-1. for b in boxes[idx_cls]])
				new_boxes = detect_fn(boxes[idx_cls].tolist(), affinity, *args, **kwargs)
			else:
				new_boxes = detect_fn(boxes[idx_cls].tolist(), *args, **kwargs)

			objs.extend(new_boxes)
		boxes = objs
	return boxes

def _greedy(boxes, *args, **kwargs):
	overlap=0.4
	if 'overlap' in kwargs:
		overlap = kwargs['overlap']
	min_box_per_region = 1
        if 'min_box_per_region' in kwargs:
                min_box_per_region = kwargs['min_box_per_region']
	if len(boxes) == 0:
		return []
	
	boxes = np.asarray(boxes)
	conf = np.asarray([b.confidence for b in boxes])
	conf_idx = np.argsort(conf)[::-1]
	boxes = boxes[conf_idx].tolist()
	used_boxes = []
	curr_box = boxes.pop(0)

	while len(boxes) > 0:
		i = 0
		n_region = 1
		while i < len(boxes):
			box = boxes[i]
			if curr_box.iou(box) > overlap:
				_ = boxes.pop(i)
				n_region += 1
			else:
				i += 1
		if n_region >= min_box_per_region:
			used_boxes.append(curr_box)
		if len(boxes) > 0:
			curr_box = boxes.pop(0)

	return used_boxes

def _viola_jones(boxes, *args, **kwargs):
	'''
	Calculate score for the combined boxes
	'''
	overlap = 0.4
	if 'overlap' in kwargs:
		overlap = kwargs['overlap']

	min_box_per_region = 1
	if 'min_box_per_region' in kwargs:
		min_box_per_region = kwargs['min_box_per_region']

	if boxes.__len__() == 0:
		return []

	# sort boxes by confidence
	sort_idx = np.argsort([box.confidence for box in boxes])[::-1]
	boxes = np.asarray(boxes)[sort_idx].tolist()

	scores = [box.confidence for box in boxes]
	regions = []
	region_scores = []
	# split boxes into disjoint sets
	while boxes.__len__() > 0:
		curr_box = boxes.pop(0)
		curr_score = scores.pop(0)

		in_region = False
		for region, score in zip(regions, region_scores):
			rlen = len(region)
			for i in range(rlen):
				box = region[i]
				in_region |= box.iou(curr_box) > overlap
				if in_region:
					region.append(curr_box)
					score.append(curr_score)
					break
			if in_region:
				break
		if not in_region:
			regions.append([curr_box])
			region_scores.append([curr_score])

	objs = []
	new_scores = []
	for region, score in zip(regions, region_scores):
		if len(region) < min_box_per_region:
			continue

		box_init = region[0] / len(region)
		box_init.confidence = np.max([score])

		# average the coordinate and class confidence scores for boxes in the same region
		for box in region[1:]:
			box_init += (box / len(region))

		# set the class name
		objs.append(box_init)

	return objs
