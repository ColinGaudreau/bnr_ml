import numpy as np
from PIL import Image, ImageDraw

import pdb

class BoundingBox(object):
	def __init__(self, xi,yi,xf,yf, label=None):
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
		self.label = label
	@property
	def w(self):
		return self.xf - self.xi
	@property
	def h(self):
		return self.yf - self.yi
	@property
	def size(self):
		return self.w*self.h

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
	def to_arr(self):
		return [self.xi, self.yi, self.xf, self.yi]
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
	def to_vec(self, size, num_classes=None):
		self.xi, self.xf = self.xi / size[1], self.xf / size[1]
		self.yi, self.yf = self.yi / size[0], self.yf / size[0]
		vec = np.asarray([self.xi, self.yi, self.xf - self.xi, self.yf - self.yi])
		if num_classes is not None:
			label = np.zeros(num_classes)
			if self.label is not None:
				label[self.label] = 1.
			vec = np.concatenate((vec, label))
		return vec
	def round(self):
		self.xi, self.yi, self.xf, self.yf = round(self.xi), round(self.yi), round(self.xf), round(self.yf)
	def __str__(self):
		return '(' + str(self.xi) + ',' + str(self.yi) + ') (' + str(self.xf) + ',' + str(self.yf) + ')'
	def __repr__(self):
		return self.__str__()
	def __mul__(self, scale):
		return BoundingBox(self.xi * scale[0], self.yi * scale[1], self.xf * scale[0], self.yf * scale[0])
	def __truediv__(self, scale):
		return BoundingBox(self.xi / scale[0], self.yi / scale[1], self.xf / scale[0], self.yf / scale[0])
	def __add__(self, shift):
		return BoundingBox(self.xi + shift[0], self.yi + shift[1], self.xf + shift[0], self.yf + shift[1])
	def __sub__(self, shift):
		return BoundingBox(self.xi - shift[0], self.yi - shift[1], self.xf - shift[0], self.yf - shift[1])

	@staticmethod
	def gen_randombox(overlap, box, eps=.9):
		vec = np.random.randn(4)
		vec /= np.sqrt(np.sum(vec**2))
		new_box = box.copy()
		while box.overlap(new_box) > overlap:
			new_box.xi += vec[0]
			new_box.yi += vec[1]
			new_box.xf += vec[2]
			new_box.yf += vec[3]
		return new_box

def transform_coord(coord, new_size, old_size, normalize=True):
	new_coord = np.copy(coord)
	if normalize:
		new_coord[[0,2]] *= 1. / new_size[1]
		new_coord[[1,3]] *= 1. / new_size[0]
	else:
		new_coord[[0,2]] *= old_size[1] / new_size[1]
		new_coord[[1,3]] *= old_size[0] / new_size[0]
	return new_coord

def nms(preds, thresh=0.3):
	if preds.shape[0] == 0:
		return preds
	idx = np.argsort(preds[:,4])
	pick = np.zeros_like(idx).astype(np.int32)
	area = np.maximum(0, (preds[:,2] - preds[:,0])) * np.maximum(0, (preds[:,3] - preds[:,1]))
	counter = 0
	while idx.size > 0:
		last = idx.size - 1
		i = idx[last]
		pick[counter] = i
		counter = counter + 1

		xi = np.maximum(preds[i,0], preds[idx[0:last - 1],0])
		xf = np.minimum(preds[i,2], preds[idx[0:last - 1],2])
		yi = np.maximum(preds[i,1], preds[idx[0:last - 1],1])
		yf = np.minimum(preds[i,3], preds[idx[0:last - 1],3])

		w, h = np.maximum(0., xf - xi), np.maximum(0., yf - yi)
		isec = w * h
		o = isec / (area[idx[last]] + area[idx[0:last - 1]] - isec)
		idx = np.delete(idx, last, 0)
		idx = idx[o<thresh]
	pick = pick[0:counter]
	return preds[pick,:]

def draw_coord(im, coords, label_map=None):
	coords = np.copy(coords)
	if im.max() <= 1:
		im = im * 255
	if im.dtype != np.uint8:
		im = im.astype(np.uint8)
	im = Image.fromarray(im)

	if coords.shape[0] == 0:
		return im

	draw = ImageDraw.Draw(im)
		
	def draw_rects(draw, size, coords, color=(255, 255, 255)):
		for i in range(coords.shape[0]):
			coord = coords[i, :4]
			coord[[0,2]] *= size[1]
			coord[[1,3]] *= size[0]
			coord[[2,3]] += coord[[0,1]]
			coord = np.int_(coord).tolist()
			draw.rectangle(coord, outline=color)

	if coords.shape[1] > 4:
		unique_classes = np.unique(coords[:,-1])
		for cls in unique_classes:
			class_idx = coords[:,-1] == cls
			class_coords = coords[class_idx]
			color = tuple(np.int_(255 * np.random.rand(3,))) # color for class
			draw_rects(draw, im.size, class_coords, color=color)
			for i in range(class_coords.shape[0]):
				coord = class_coords[i]
				text = 'confidence: %.2f' % class_coords[i, -2]
				if label_map is not None:
					text = '%s, %s' % (label_map(class_coords[i,-1]), text)
				draw.text(coord[:2].tolist(), text, fill=color)
		return im
	else:
		draw_rects(draw, im.size, coords)
		return im





















