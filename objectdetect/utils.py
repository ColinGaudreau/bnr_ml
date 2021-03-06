import numpy as np
import numpy.random as npr
from PIL import Image, ImageDraw
import theano
import theano.tensor as T
from bnr_ml.utils.helpers import meshgrid

import pdb

class BoundingBox(object):
	'''
	Helper class for managing bounding boxes.

	Parameters
	----------
	xi : float
		x-coordinate of anchor point.
	yi : float
		y-coordinate of anchor point.
	xf : float
		x-coordinate of the box end.
	yf : float
		y-coordinate of the box end.
	cls : str (default '')
		Class of object.
	confidence : float (default 0.)
		Confidence in detection.
	'''
	def __init__(self, xi,yi,xf,yf, cls='', confidence=0.):
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
		'''Width of box.'''
		return self.xf - self.xi
	@property
	def h(self):
		'''Height of box.'''
		return self.yf - self.yi
	@property
	def size(self):
		'''Area of box.'''
		return self.w*self.h

	def __setattr__(self, name, value):
		if name == 'w':
			self.xf = self.xi + value
		elif name == 'h':
			self.yf = self.yi + value
		else:
			return super(BoundingBox, self).__setattr__(name, value)

	def iou(self, box):
		'''
		Intersection-over-union of box with another box.

		Parameters
		----------
		box : :class:`BoundingBox` instance
			Box with which to calculate IOU.
		'''
		isec = self.intersection(box)
		union = self.size + box.size - isec.size
		if union > 0:
			return isec.size / union
		else:
			return 0.
	def overlap(self, box):
		'''
		Find area which corresponds to overlap with another box.

		Parameters
		----------
		box : :class:`BoundingBox` instance
			Box with which to find overlap.
		'''
		if self.size > 0:
			return self.intersection(box).size / self.size
		else:
			return 0.

	def intersection(self, box):
		'''
		Find box which corresponds to intersection with another box.

		Parameters
		----------
		box : :class:`BoundingBox` instance
			Box with which to find intersection.
		'''
		new_xi = max(self.xi, box.xi)
		new_yi = max(self.yi, box.yi)
		new_xf = min(self.xf, box.xf)
		new_yf = min(self.yf, box.yf)
		if new_xi > new_xf or new_yi > new_yf:
			new_xi, new_yi, new_xf, new_yf = 0., 0., 0., 0.
		return BoundingBox(new_xi, new_yi, new_xf, new_yf)

	def tolist(self):
		'''Get bounding box parameters as a `list`.'''
		return [self.xi, self.yi, self.xf, self.yf]

	def tondarray(self):
		'''Get bounding box parameters as `numpy.ndarray`, however it's parametrized using w,h rather than xf, yf.'''
		return np.asarray([self.xi, self.yi, self.w, self.h])

	def isvalid(self):
		'''Ensure bounding box parameters correspond to valid box.'''
		valid = True
		valid = valid and self.w > 0 and self.h > 0
		valid = valid and self.xf >= self.xi
		valid = valid and self.yf >= self.yi
		return valid

	def copy(self):
		'''Copy box.'''
		return BoundingBox(self.xi, self.yi, self.xf, self.yf)

	def subimage(self, im):
		'''
		Returns portion of the image corresponding to the bounding box.
		
		Parameters
		----------
		im : numpy.ndarray
			Input image.
		'''
		xi = max(0, self.xi)
		yi = max(0, self.yi)
		xf = min(im.shape[1], self.xf)
		yf = min(im.shape[0], self.yf)
		xi, yi, xf, yf = int(xi), int(yi), int(xf), int(yf)
		return im[yi:yf, xi:xf,:]

	def round(self):
		self.xi, self.yi, self.xf, self.yf = round(self.xi), round(self.yi), round(self.xf), round(self.yf)


	def __str__(self):
		if self.cls == '':
			retstr = 'BoundingBox([{},{},{},{}])'.format(self.xi, self.yi, self.xf, self.yf)
		else:
			retstr = 'BoundingBox([{},{},{},{}], class={}, confidence={})'.format(self.xi, self.yi, self.xf, self.yf, self.cls, self.confidence)
		return retstr

	def __repr__(self):
		return self.__str__()

	def __mul__(self, other):
		if other.__class__ == tuple:
			return BoundingBox(self.xi * other[1], self.yi * other[0], self.xf * other[1], self.yf * other[0], cls=self.cls, confidence=self.confidence)
		else:
			return BoundingBox(self.xi * other, self.yi * other, self.xf * other, self.yf * other, cls=self.cls, confidence=self.confidence)

	def __rmul__(self, other):
		if other.__class__ == tuple:
			return BoundingBox(self.xi * other[1], self.yi * other[0], self.xf * other[1], self.yf * other[0], cls=self.cls, confidence=self.confidence)
		else:
			return BoundingBox(self.xi * other, self.yi * other, self.xf * other, self.yf * other, cls=self.cls, confidence=self.confidence)

	def __imul__(self, other):
		if other.__class__ == tuple:
			return BoundingBox(self.xi * other[1], self.yi * other[0], self.xf * other[1], self.yf * other[0], cls=self.cls, confidence=self.confidence)
		else:
			return BoundingBox(self.xi * other, self.yi * other, self.xf * other, self.yf * other, cls=self.cls, confidence=self.confidence)

	def __div__(self, other):
		if other.__class__ == tuple:
			return BoundingBox(self.xi / other[1], self.yi / other[0], self.xf / other[1], self.yf / other[0], cls=self.cls, confidence=self.confidence)
		else:
			return BoundingBox(self.xi / other, self.yi / other, self.xf / other, self.yf / other, cls=self.cls, confidence=self.confidence)

	def __rdiv__(self, other):
		if other.__class__ == tuple:
			return BoundingBox(self.xi / other[1], self.yi / other[0], self.xf / other[1], self.yf / other[0], cls=self.cls, confidence=self.confidence)
		else:
			return BoundingBox(self.xi / other, self.yi / other, self.xf / other, self.yf / other, cls=self.cls, confidence=self.confidence)

	def __idiv__(self, other):
		if other.__class__ == tuple:
			return BoundingBox(self.xi / other[1], self.yi / other[0], self.xf / other[1], self.yf / other[0], cls=self.cls, confidence=self.confidence)
		else:
			return BoundingBox(self.xi / other, self.yi / other, self.xf / other, self.yf / other, cls=self.cls, confidence=self.confidence)

	def __add__(self, other):
		if other.__class__ != BoundingBox:
			return BoundingBox(self.xi + other, self.yi + other, self.xf + other, self.yf + other, cls=self.cls, confidence=self.confidence)
		else:
			return BoundingBox(self.xi + other.xi, self.yi + other.yi, self.xf + other.xf, self.yf + other.yf, cls=self.cls, confidence=self.confidence)

	def __radd__(self, other):
		if other.__class__ != BoundingBox:
			return BoundingBox(self.xi + other, self.yi + other, self.xf + other, self.yf + other, cls=self.cls, confidence=self.confidence)
		else:
			return BoundingBox(self.xi + other.xi, self.yi + other.yi, self.xf + other.xf, self.yf + other.yf, cls=self.cls, confidence=self.confidence)

	def __iadd__(self, other):
		if other.__class__ != BoundingBox:
			return BoundingBox(self.xi + other, self.yi + other, self.xf + other, self.yf + other, cls=self.cls, confidence=self.confidence)
		else:
			return BoundingBox(self.xi + other.xi, self.yi + other.yi, self.xf + other.xf, self.yf + other.yf, cls=self.cls, confidence=self.confidence)

	def __sub__(self, other):
		if other.__class__ != BoundingBox:
			return BoundingBox(self.xi - other, self.yi - other, self.xf - other, self.yf - other, cls=self.cls, confidence=self.confidence)
		else:
			return BoundingBox(self.xi - other.xi, self.yi - other.yi, self.xf - other.xf, self.yf - other.yf, cls=self.cls, confidence=self.confidence)

	def __rsub__(self, other):
		if other.__class__ != BoundingBox:
			return BoundingBox(self.xi - other, self.yi - other, self.xf - other, self.yf - other, cls=self.cls, confidence=self.confidence)
		else:
			return BoundingBox(self.xi - other.xi, self.yi - other.yi, self.xf - other.xf, self.yf - other.yf, cls=self.cls, confidence=self.confidence)

	def __isub__(self, other):
		if other.__class__ != BoundingBox:
			return BoundingBox(self.xi - other, self.yi - other, self.xf - other, self.yf - other, cls=self.cls, confidence=self.confidence)
		else:
			return BoundingBox(self.xi - other.xi, self.yi - other.yi, self.xf - other.xf, self.yf - other.yf, cls=self.cls, confidence=self.confidence)

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

def draw_boxes(im, boxes, color=(255,255,255)):
	'''
	Draw boxes on image from list of :class:`BoundingBox` instances.

	Parameters
	----------
	im : numpy.ndarray
		blah
	boxes : list
		blah
	'''
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
	colors = {}
	unique_labels = np.unique([box.cls for box in boxes]).tolist()
	if isinstance(color, tuple):
		for label in unique_labels:
			colors[label] = tuple(np.int_(150 + 105 * npr.rand(3,)))
	elif isinstance(color, dict):
		colors = color

	for box in boxes:
		if box.cls is not None:
			label_color = colors[box.cls]
			text = str(box.cls)
		else:
			label_color = color
			text = ''

		if box.confidence > 0:
			if box.cls is not None:
				text += ', '
			text += ('Confidence: %.2f' % box.confidence)
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

def iou_matrix(preds):
	'''
	Calculate iou matrix for a list of predictions.

	Parameters
	----------
	preds : theano.tensor
		:math:`N \\times 4` `theano.tensor` list of bounding box parameters parameterized as :math:`(x_i, y_i, x_f, y_f)`.

	Returns
	-------
	theano.tensor
		Matrix of IOU values.
	'''
	idx1, idx2 = meshgrid(T.arange(preds.shape[0]), T.arange(preds.shape[0]))
	preds1, preds2 = preds[idx1,:], preds[idx2,:]
	
	xi, yi = T.maximum(preds1[:,:,0], preds2[:,:,0]), T.maximum(preds1[:,:,1], preds2[:,:,1])
	xf, yf = T.minimum(preds1[:,:,2], preds2[:,:,2]), T.minimum(preds1[:,:,3], preds2[:,:,3])

	w, h = T.maximum(xf - xi, 0.) ,T.maximum(yf - yi, 0.)

	isec = w * h
	u = (preds1[:,:,2]-preds1[:,:,0]) * (preds1[:,:,3]-preds1[:,:,1]) + (preds2[:,:,2]-preds2[:,:,0]) * (preds2[:,:,3]-preds2[:,:,1]) - isec

	return isec / u



