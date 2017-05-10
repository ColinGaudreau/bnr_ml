from bnr_ml.objectdetect.utils import BoundingBox
import copy

def nms(boxes, *args, **kwargs):
	'''
	Takes list of BoundingBox objects and does non maximal suppression.
	'''
	boxes = copy.deepcopy(boxes)
	classes = np.unique([box.cls for box in boxes])
	objs = []
	# nms for each class
	for cls in classes:
		boxes_per_cls = np.asarray([box for box in boxes if box.cls == cls])
		objs.extend(_viola_jones(boxes, *args, **kwargs))
	return objs

def _viola_jones(boxes, overlap=0.4):
	'''
	Calculate score for the combined boxes
	'''
	if boxes.__len__() == 0:
		return []
	assert(boxes.__len__() == scores.__len__())
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
		box_init = region[0] / len(region)

		# average the coordinate and class confidence scores for boxes in the same region
		for box, s in zip(region[1:], score[1:]):
			box_init += (box / len(region))
			box_init.confidence += (s / len(score))

		# set the class name
		box_init.cls = boxes[0].cls
		objs.append(box_init)

	return objs
