import numpy as np

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
	def gen_randombox(iou, box, eps=.9):
		angle = 2 * np.pi * np.random.rand()
		delx, dely = eps*np.cos(angle), eps*np.sin(angle)
		new_box = box.copy()
		while new_box.iou(box) > iou:
			new_box.xi += delx
			new_box.yi += dely
			new_box.xf += delx
			new_box.yf += dely
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

def nms(preds, thresh):
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
