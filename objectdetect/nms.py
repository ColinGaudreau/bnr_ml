from bnr_ml.objectdetect.utils import BoundingBox

def nms(boxes, *args, **kwargs):
	'''
	Takes list of BoundingBox objects and does non maximal suppression.
	'''
	return _viola_jones(boxes, *args, **kwargs)

def _viola_jones(boxes, overlap=0.4):
	'''
	Non maximal suppression algorithm described in the Viola-Jones paper.
	'''
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