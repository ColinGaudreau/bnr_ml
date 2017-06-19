from bnr_ml.objectdetect.utils import BoundingBox
import copy

METHOD_VIOLA_JONES = 'viola-jones'
METHOD_GREEDY = 'greedy'

def nms(boxes, *args, **kwargs):
	'''
	Takes list of BoundingBox objects and does non maximal suppression.
	'''
	# nms for each class
	n_apply = 1
	if 'n_apply' in kwargs:
		n_apply = kwargs['n_apply']

	classes = list(set([box.cls for box in boxes]))
	boxes = copy.deepcopy(boxes)

	method = METHOD_VIOLA_JONES
	if 'method' in kwargs:
		method = kwargs['method']
	if method.lower() == METHOD_VIOLA_JONES:
		detect_fn = lambda boxes, *args, **kwargs: _viola_jones(boxes, *args, **kwargs)
	elif method.lower() == METHOD_GREEDY:
		detect_fn = lambda boxes, *args, **kwargs: _greedy(boxes, *args, **kwargs)
	else:
		raise Exception('Method "{}" not valid.'.format(method))

	for _ in range(n_apply):
		objs = []
		for cls in classes:
			boxes_per_cls = [box for box in boxes if box.cls == cls]
			objs.extend(_viola_jones(boxes_per_cls, *args, **kwargs))
		boxes = objs
	return boxes

def _greedy(boxes, *args, **kwargs):
	overlap=0.4
	if 'overlap' in kwargs:
		overlap = kwargs['overlap']

	boxes = np.asarray(boxes)
	conf = np.asarray([b.confidence for b in boxes])
	conf_idx = np.argsort(conf)[::-1]
	boxes = boxes[conf_idx].tolist()
	used_boxes = []
	curr_box = boxes.pop(0)

	while len(boxes) > 0:
		i = 0
		while i < len(boxes):
			box = boxes[i]
			if curr_box.iou(box) > overlap:
				_ = boxes.pop(i)
			else:
				i += 1
		used_boxes.append(curr_box)
		curr_box = boxes.pop(0)

	return used_boxes

def _viola_jones(boxes, *args, **kwargs):
	'''
	Calculate score for the combined boxes
	'''
	overlap = 0.4
	if 'overlap' in kwargs:
		overlap = kwargs['overlap']

	min_box_per_region = 2
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
