from bnr_ml.objectdetect.utils import BoundingBox
import copy

def nms(boxes, *args, **kwargs):
	'''
	Takes list of BoundingBox objects and does non maximal suppression.
	'''
	box = copy.copy(boxes)
	return _viola_jones(boxes, *args, **kwargs)

def _viola_jones(boxes, scores=None, overlap=0.4):
	'''
	Non maximal suppression algorithm described in the Viola-Jones paper.
	'''
	if score is not None:
		return _viola_jones_with_scores(boxes, scores, overlap=overlap)

	regions = []
	# split boxes into disjoint sets
	while boxes.__len__() > 0:
		curr_box = boxes.pop(0)
		in_region = False
		for region in regions:
			rlen = len(region)
			for i in range(rlen):
				box = region[i]
				in_region |= box.iou(curr_box) > overlap
				if in_region:
					region.append(curr_box)
					break
			if in_region:
				break
		if not in_region:
			regions.append([curr_box])
	objs = []
	for region in regions:
		box_init = region[0]
		box_init = box_init / len(region)
		for box in region[1:]:
			box_init = box_init + (box / len(region))
		objs.append(box_init)
	return objs

def _viola_jones_with_scores(boxes, scores, overlap=0.4):
	'''
	Calculate score for the combined boxes
	'''
	assert(boxes.__len__() == scores.__len__())
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
					region_scores.append(curr_score)
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
		score_init = score[0] / len(score)

		for box, s in zip(region[1:], score[1:]):
			box_init = box_init + (box / len(region))
			box_init += (box / len(region))
			score_init += (s / len(score))

		objs.append(box_init)
		new_scores.append(score_init)

	return objs, new_scores
