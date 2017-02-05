import numpy as np
import theano
from theano import tensor as T
import lasagne
import os
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize, rotate
import pdb

class StreamPrinter(object):
	def __init__(self, *args):
		self.streams = args

	def println(self, str):
		for stream in self.streams:
			stream.write('%s\n' % str)

	def close(self):
		for stream in self.streams:
			stream.close()

def save_all_layers(network_output, dirname):
	layers = lasagne.layers.get_all_layers(network_output)
	lcnt = 1
	for layer in layers:
		wcnt = 1
		weights = lasagne.layers.get_all_param_values(layer)
		for weight in weights:
			filename = '{}/layer{}_weight{}.npy'.format(dirname, lcnt, wcnt)
			np.save(filename, weight)
			wcnt += 1
		lcnt += 1

def meshgrid2D(arr1, arr2):
	arr1 = arr1.dimshuffle('x',0)
	arr2 = arr2.dimshuffle(0,'x')
	arr1, arr2 = T.repeat(arr1, arr2.shape[0], axis=0), T.repeat(arr2, arr1.shape[1], axis=1)
	return arr1, arr2

def meshgrid(*xi, **kwargs):
	assert(xi.__len__() > 1)
	if 'flatten' in kwargs:
		flatten = kwargs['flatten']
	else:
		flatten = False
	arrs = []
	for i, x in enumerate(xi):
		ds_args = ['x' if k != i else 0 for k in range(xi.__len__())]
		arr = x.dimshuffle(*ds_args)

		for j in range(xi.__len__()):
			if j != i:
				arr = T.repeat(arr, xi[j].shape[0], axis=j)
		if flatten:
			arr = arr.reshape((-1,))
		arrs.append(arr)

	return arrs

def softmax(mat, axis=1):
	'''	
	axis along which to take soft max, axis \in {0,1,2,3}

	Safe softmax function:
	log f(x) = x - (x_1 + log(1 + sum_{i=2}^N exp(x_i - x_1)))
	'''
	max_el = mat.max(axis=axis, keepdims=True)
	logsoftmax = mat - (max_el + T.log(T.sum(T.exp(mat - max_el), axis=axis, keepdims=True)))
	return T.exp(logsoftmax)

def load_all_layers(network_output, dirname):
	layers = lasagne.layers.get_all_layers(network_output)
	lcnt = 1
	for layer in layers:
		wcnt = 1
		num_weights = len(lasagne.layers.get_all_params(layer))
		weights = []
		for num in range(num_weights):
			filename = '{}/layer{}_weight{}.npy'.format(dirname, lcnt, wcnt)
			weights.append(np.load(filename))
			wcnt += 1
		lasagne.layers.set_all_param_values(layer,weights)
		lcnt += 1

def bitwise_not(mat):
	idx_true = T.eq(mat, 1)
	idx_false = T.eq(mat, 0)
	mat = T.set_subtensor(mat[idx_true.nonzero()], 0)
	mat = T.set_subtensor(mat[idx_false.nonzero()], 1)
	return mat

def load_bee_images_from_list(image_list,size=(200,200),color=True):
	channels = 3 if color else 1
	X = np.zeros((len(image_list),channels) + size, dtype=theano.config.floatX)
	for idx, image_name in enumerate(image_list):
		if os.path.exists(image_name):
			image = imread(image_name)
			if not color:
				image = rgb2gray(image).reshape(200,200)
			if size != image.shape[:2]:
				image = resize(image,size)
			if channels == 3:
				image = np.swapaxes(np.swapaxes(image,1,2),0,1).astype(theano.config.floatX)
			else:
				image = np.reshape(image,(1,) + size).astype(theano.config.floatX)
		else:
			raise Exception('Couldn\'t find image with name "{}"'.format(image_name))
		if image.shape[0] == 4:
			image = image[:3]
		X[idx] = image
	return X

def load_bee_image(imname, size=(200,200), color=True, angle=0):
	if os.path.exists(imname):
		im = imread(imname)
		mindim = min(*im.shape[:2])
		im = im[:mindim, :mindim]
		if not color:
			im = rgb2gray(im)
		if color and im.shape[2] == 4:
			im = im[:,:,:3]
		im = resize(im, size)
		return rotate(im,angle)
	else:
		raise Exception('Couldn\'t find image with name "{}"'.format(image_name))

def load_bee_images(imlist, size=(200,200), color=True, angle_list=None):
	channels = 3 if color else 1
	X = np.zeros((len(imlist),channels,) + size, dtype=theano.config.floatX)
	for idx, imname in enumerate(imlist):
		if angle_list is not None:
			angle = angle_list[idx]
		else:
			angle = 0
		im = load_bee_image(imname, size, color, angle)
		if not color:
			im = im.reshape((1,) + size)
		else:
			im = im.swapaxes(2,1).swapaxes(1,0)
		X[idx] = im.astype(theano.config.floatX)
	return X

def scale_to_unit_interval(ndar, eps=1e-8):
	""" Scales all values in the ndarray ndar to be between 0 and 1 """
	ndar = ndar.copy()
	ndar -= ndar.min()
	ndar *= 1.0 / (ndar.max() + eps)
	return ndar

def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
					   scale_rows_to_unit_interval=True,
					   output_pixel_vals=True):
	"""
	Transform an array with one flattened image per row, into an array in
	which images are reshaped and layed out like tiles on a floor.

	This function is useful for visualizing datasets whose rows are images,
	and also columns of matrices for transforming those rows
	(such as the first layer of a neural net).

	:type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
	be 2-D ndarrays or None;
	:param X: a 2-D array in which every row is a flattened image.

	:type img_shape: tuple; (height, width)
	:param img_shape: the original shape of each image

	:type tile_shape: tuple; (rows, cols)
	:param tile_shape: the number of images to tile (rows, cols)

	:param output_pixel_vals: if output should be pixel values (i.e. int8
	values) or floats

	:param scale_rows_to_unit_interval: if the values need to be scaled before
	being plotted to [0,1] or not


	:returns: array suitable for viewing as an image.
	(See:`Image.fromarray`.)
	:rtype: a 2-d array with same dtype as X.

	"""

	assert len(img_shape) == 2
	assert len(tile_shape) == 2
	assert len(tile_spacing) == 2

	# The expression below can be re-written in a more C style as
	# follows :
	#
	# out_shape	= [0,0]
	# out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
	#				tile_spacing[0]
	# out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
	#				tile_spacing[1]
	out_shape = [
		(ishp + tsp) * tshp - tsp
		for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
	]

	if isinstance(X, tuple):
		assert len(X) == 4
		# Create an output np ndarray to store the image
		if output_pixel_vals:
			out_array = np.zeros((out_shape[0], out_shape[1], 4),
									dtype='uint8')
		else:
			out_array = np.zeros((out_shape[0], out_shape[1], 4),
									dtype=X.dtype)

		#colors default to 0, alpha defaults to 1 (opaque)
		if output_pixel_vals:
			channel_defaults = [0, 0, 0, 255]
		else:
			channel_defaults = [0., 0., 0., 1.]

		for i in range(4):
			if X[i] is None:
				# if channel is None, fill it with zeros of the correct
				# dtype
				dt = out_array.dtype
				if output_pixel_vals:
					dt = 'uint8'
				out_array[:, :, i] = np.zeros(
					out_shape,
					dtype=dt
				) + channel_defaults[i]
			else:
				# use a recurrent call to compute the channel and store it
				# in the output
				out_array[:, :, i] = tile_raster_images(
					X[i], img_shape, tile_shape, tile_spacing,
					scale_rows_to_unit_interval, output_pixel_vals)
		return out_array

	else:
		# if we are dealing with only one channel
		H, W = img_shape
		Hs, Ws = tile_spacing

		# generate a matrix to store the output
		dt = X.dtype
		if output_pixel_vals:
			dt = 'uint8'
		out_array = np.zeros(out_shape, dtype=dt)

		for tile_row in range(tile_shape[0]):
			for tile_col in range(tile_shape[1]):
				if tile_row * tile_shape[1] + tile_col < X.shape[0]:
					this_x = X[tile_row * tile_shape[1] + tile_col]
					if scale_rows_to_unit_interval:
						# if we should scale values to be between 0 and 1
						# do this by calling the `scale_to_unit_interval`
						# function
						this_img = scale_to_unit_interval(
							this_x.reshape(img_shape))
					else:
						this_img = this_x.reshape(img_shape)
					# add the slice to the corresponding position in the
					# output array
					c = 1
					if output_pixel_vals:
						c = 255
					out_array[
						tile_row * (H + Hs): tile_row * (H + Hs) + H,
						tile_col * (W + Ws): tile_col * (W + Ws) + W
					] = this_img * c
		return out_array
















