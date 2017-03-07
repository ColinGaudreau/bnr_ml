import numpy as np
import numpy.random as npr
from PIL import Image, ImageDraw

import pdb

class BoundingBox(object):
	'''
	Helper class for managing bounding boxes
	'''
	def __init__(self, xi,yi,xf,yf, cls=None, confidence=-1.):
		if xi > xf:
			self.xi = xf
			self.xf = xi
		else:
			self.xi = xi
			self.xf = xf
		if yi > yf:
			self.yi = yf
			self.yf = yi
		else:
			self.yi = yi
			self.yf = yf
		self.cls = cls
		self.confidence = confidence

	@property
	def w(self):
		return self.xf - self.xi
	@property
	def h(self):
		return self.yf - self.yi
	@property
	def size(self):
		return self.w*self.h

	def __setattr__(self, name, value):
		if name == 'w':
			self.xf = self.xi + value
		elif name == 'h':
			self.yf = self.yi + value
		else:
			return super(BoundingBox, self).__setattr__(name, value)

	def iou(self, box):
		isec = self.intersection(box)
		union = self.size + box.size - isec.size
		if union > 0:
			return isec.size / union
		else:
			return 0.
	def overlap(self, box):
		if self.size > 0:
			return self.intersection(box).size / self.size
		else:
			return 0.

	def intersection(self, box):
		new_xi = max(self.xi, box.xi)
		new_yi = max(self.yi, box.yi)
		new_xf = min(self.xf, box.xf)
		new_yf = min(self.yf, box.yf)
		if new_xi > new_xf or new_yi > new_yf:
			new_xi, new_yi, new_xf, new_yf = 0., 0., 0., 0.
		return BoundingBox(new_xi, new_yi, new_xf, new_yf)

	def tolist(self):
		return [self.xi, self.yi, self.xf, self.yf]

	def tondarray(self):
		return np.asarray([self.xi, self.yi, self.w, self.h])

	def isvalid(self):
		valid = True
		valid = valid and self.w > 0 and self.h > 0
		valid = valid and self.xf >= self.xi
		valid = valid and self.yf >= self.yi
		return valid

	def copy(self):
		return BoundingBox(self.xi, self.yi, self.xf, self.yf)

	def subimage(self, im):
		xi = max(0, self.xi)
		yi = max(0, self.yi)
		xf = min(im.shape[1], self.xf)
		yf = min(im.shape[0], self.yf)
		return im[yi:yf, xi:xf,:]

	def round(self):
		self.xi, self.yi, self.xf, self.yf = round(self.xi), round(self.yi), round(self.xf), round(self.yf)

	'''
	Override operator for easier use of BoundingBox class
	'''

	def __str__(self):
		return 'BoundingBox([' + str(self.xi) + ',' + str(self.yi) + ',' + str(self.xf) + ',' + str(self.yf) + '])'

	def __repr__(self):
		return self.__str__()

	def __mul__(self, other):
		if other.__class__ == tuple:
			return BoundingBox(self.xi * other[1], self.yi * other[0], self.xf * other[1], self.yf * other[0])
		else:
			return BoundingBox(self.xi * other, self.yi * other, self.xf * other, self.yf * other)

	def __rmul__(self, other):
		if other.__class__ == tuple:
			return BoundingBox(self.xi * other[1], self.yi * other[0], self.xf * other[1], self.yf * other[0])
		else:
			return BoundingBox(self.xi * other, self.yi * other, self.xf * other, self.yf * other)

	def __imul__(self, other):
		if other.__class__ == tuple:
			return BoundingBox(self.xi * other[1], self.yi * other[0], self.xf * other[1], self.yf * other[0])
		else:
			return BoundingBox(self.xi * other, self.yi * other, self.xf * other, self.yf * other)

	def __div__(self, other):
		if other.__class__ == tuple:
			return BoundingBox(self.xi / other[1], self.yi / other[0], self.xf / other[1], self.yf / other[0])
		else:
			return BoundingBox(self.xi / other, self.yi / other, self.xf / other, self.yf / other)

	def __rdiv__(self, other):
		if other.__class__ == tuple:
			return BoundingBox(self.xi / other[1], self.yi / other[0], self.xf / other[1], self.yf / other[0])
		else:
			return BoundingBox(self.xi / other, self.yi / other, self.xf / other, self.yf / other)

	def __idiv__(self, other):
		if other.__class__ == tuple:
			return BoundingBox(self.xi / other[1], self.yi / other[0], self.xf / other[1], self.yf / other[0])
		else:
			return BoundingBox(self.xi / other, self.yi / other, self.xf / other, self.yf / other)

	def __add__(self, other):
		if other.__class__ != BoundingBox:
			return BoundingBox(self.xi + other, self.yi + other, self.xf + other, self.yf + other)
		else:
			return BoundingBox(self.xi + other.xi, self.yi + other.yi, self.xf + other.xf, self.yf + other.yf)

	def __radd__(self, other):
		if other.__class__ != BoundingBox:
			return BoundingBox(self.xi + other, self.yi + other, self.xf + other, self.yf + other)
		else:
			return BoundingBox(self.xi + other.xi, self.yi + other.yi, self.xf + other.xf, self.yf + other.yf)

	def __iadd__(self, other):
		if other.__class__ != BoundingBox:
			return BoundingBox(self.xi + other, self.yi + other, self.xf + other, self.yf + other)
		else:
			return BoundingBox(self.xi + other.xi, self.yi + other.yi, self.xf + other.xf, self.yf + other.yf)

	def __sub__(self, other):
		if other.__class__ != BoundingBox:
			return BoundingBox(self.xi - other, self.yi - other, self.xf - other, self.yf - other)
		else:
			return BoundingBox(self.xi - other.xi, self.yi - other.yi, self.xf - other.xf, self.yf - other.yf)

	def __rsub__(self, other):
		if other.__class__ != BoundingBox:
			return BoundingBox(self.xi - other, self.yi - other, self.xf - other, self.yf - other)
		else:
			return BoundingBox(self.xi - other.xi, self.yi - other.yi, self.xf - other.xf, self.yf - other.yf)

	def __isub__(self, other):
		if other.__class__ != BoundingBox:
			return BoundingBox(self.xi - other, self.yi - other, self.xf - other, self.yf - other)
		else:
			return BoundingBox(self.xi - other.xi, self.yi - other.yi, self.xf - other.xf, self.yf - other.yf)

	@staticmethod
	def gen_randombox(overlap, box, eps=.9):
		vec = npr.randn(4)
		vec /= np.sqrt(np.sum(vec**2))
		new_box = box.copy()
		while box.iou(new_box) > overlap:
			new_box.xi += vec[0]
			new_box.yi += vec[1]
			new_box.xf += vec[2]
			new_box.yf += vec[3]
		return new_box

def draw_boxes(im, boxes, class_scores=None, class_labels=None, color=(255,255,255)):
	if not isinstance(im, Image.Image):
		if im.max() <= 1:
			im = im * 255
		if im.dtype != np.uint8:
			im = im.astype(np.uint8)
		im = Image.fromarray(im)

	if boxes.__len__() < 1:
		return im

	draw = ImageDraw.Draw(im)

	# decide color for labels
	if class_labels is not None and isinstance(color, tuple):
		color = {}
		unique_labels = np.unique(class_labels).tolist()
		for label in unique_labels:
			color[label] = tuple(np.int_(255 * npr.rand(3,)))

	for i, box in enumerate(boxes):
		if class_labels is not None:
			label_color = color[class_labels[i]]
			text = class_labels[i]
		else:
			label_color = color
			text = ''

		if class_scores is not None:
			if class_labels is not None:
				text += ', '
			text += ('Confidence: %.2f' % class_scores[i])

		draw.rectangle(box.tolist(), outline=label_color)
		draw.text(box.tolist()[:2], text, fill=label_color)

	return im

def _draw_rects(draw, size, coords, color=(255, 255, 255)):
		for i in range(coords.shape[0]):
			coord = coords[i, :4]
			coord[[0,2]] *= size[0]
			coord[[1,3]] *= size[1]
			coord[[2,3]] += coord[[0,1]]
			coord = np.int_(coord).tolist()
			draw.rectangle(coord, outline=color)

def draw_coord(im, coords, color=(255,255,255), label_map=None):
	coords = np.copy(coords)
	if im.max() <= 1:
		im = im * 255
	if im.dtype != np.uint8: 
		im = im.astype(np.uint8)
	im = Image.fromarray(im)

	if coords.shape[0] == 0:
		return im

	draw = ImageDraw.Draw(im)

	if coords.shape[1] > 4:
		unique_classes = np.unique(coords[:,-1])
		for cls in unique_classes:
			class_idx = coords[:,-1] == cls
			class_coords = coords[class_idx]
			color = tuple(np.int_(255 * npr.rand(3,))) # color for class
			_draw_rects(draw, im.size, class_coords, color=color)
			for i in range(class_coords.shape[0]):
				coord = class_coords[i]
				text = 'confidence: %.2f' % class_coords[i, -2]
				if label_map is not None:
					text = '%s, %s' % (label_map(class_coords[i,-1]), text)
				draw.text(coord[:2].tolist(), text, fill=color)
		return im
	else:
		_draw_rects(draw, im.size, coords, color=color)
		return im



