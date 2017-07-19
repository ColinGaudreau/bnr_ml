import theano
import theano.gpuarray.basic_ops as basic_ops
import pygpu.gpuarray as pygpu
import theano.tensor as T
from lasagne.layers import Layer

import numpy as np
import numpy.random as npr

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import pdb

roi_code = """
	#include <stdio.h>
	
	struct asg3 {
		int dim1; int dim2; int dim3;
	}; 
	
	struct shape3 {
		int dim1, __padding1; int dim2, __padding2; int dim3, __padding3;
	};
	
	struct asg4 {
		int dim1; int dim2; int dim3; int dim4;
	};
	
	struct shape4 {
		int dim1, __padding1; int dim2, __padding2; int dim3, __padding3; int dim4, __padding4;
	};
	
	struct tens3 {
		shape3 shp;
		float *data;
	};
	
	struct tens4 {
		shape4 shp;
		float *data;
	};
	
	__device__ int asg_to_ind3(asg3 asg, shape3 shp)
	{
		int ind;
		ind = (asg.dim1 * shp.dim2 * shp.dim3) +
			(asg.dim2 * shp.dim3) + asg.dim3;
		return ind;
	}
	
	__device__ int asg_to_ind4(asg4 asg, shape4 shp)
	{
		int ind;
		ind = (asg.dim1 * shp.dim2 * shp.dim3 * shp.dim4) + 
			(asg.dim2 * shp.dim3 * shp.dim4) +
			(asg.dim3 * shp.dim4) + asg.dim4;
		return ind;
	}
	
	__global__ void roi_pool(tens4 *x_pool, tens4 *x, tens3 *boxes)
	{
		// declare data here
		int r_ind, c_ind, channel, n, m, ind, i, j;
		float max_val, xi, yi, xf, yf;
		asg3 a3; asg4 a4;
		int r_min, c_min, r_max, c_max;
		
		// get indices
		r_ind = blockIdx.x; c_ind = blockIdx.y; channel = threadIdx.x;
		n = floorf(((float)blockIdx.z) / boxes->shp.dim2); m = blockIdx.z % boxes->shp.dim2;
	
		// get index for box
		a3.dim1 = n; a3.dim2 = m;
		a3.dim3 = 0; xi = floorf(x->shp.dim4 * boxes->data[asg_to_ind3(a3, boxes->shp)]);
		a3.dim3 = 1; yi = floorf(x->shp.dim3 * boxes->data[asg_to_ind3(a3, boxes->shp)]);
		a3.dim3 = 2; xf = ceilf(x->shp.dim4 * boxes->data[asg_to_ind3(a3, boxes->shp)]);
		a3.dim3 = 3; yf = ceilf(x->shp.dim3 * boxes->data[asg_to_ind3(a3, boxes->shp)]);
		
		// define column/rows to loop over
		r_min = yi + floorf((yf - yi) * float(r_ind) / x_pool->shp.dim3);
		c_min = xi + floorf((xf - xi) * float(c_ind) / x_pool->shp.dim4);
		r_max = yi + min(ceilf((yf - yi) * float(r_ind + 1) / x_pool->shp.dim3), yf - yi);
		c_max = xi + min(ceilf((xf - xi) * float(c_ind + 1) / x_pool->shp.dim4), xf - xi);
		
		a4.dim1 = n; a4.dim2 = channel; a4.dim3 = r_min; a4.dim4 = c_min;
		max_val = x->data[asg_to_ind4(a4, x->shp)];
		for(i=r_min; i<r_max; i++)
		{
			for(j=c_min; j<c_max; j++)
			{
				a4.dim3 = i; a4.dim4 = j;
				ind = asg_to_ind4(a4, x->shp);
				if(x->data[ind] > max_val)
					max_val = x->data[ind];
			}
		}
		a4.dim1 = blockIdx.z; a4.dim2 = channel; a4.dim3 = r_ind; a4.dim4 = c_ind;
		x_pool->data[asg_to_ind4(a4, x_pool->shp)] = max_val;
	}
	
	__global__ void roi_pool_grad(tens4 *x_pool_grad, tens4 *x, tens3 *boxes, tens4 *grad)
	{
		// declare data here
		int channel, n, m, ind, i, j, k, l;
		float max_val, xi, yi, xf, yf;
		asg3 a3; asg4 a4;
		int r_min, c_min, r_max, c_max, max_ind;
		
		// get indices		
		n = blockIdx.x; channel = blockIdx.y;
				
		for(m=0; m < boxes->shp.dim2; m++)
		{
			// get box
			a3.dim1 = n; a3.dim2 = m;
			a3.dim3 = 0; xi = floorf(x->shp.dim4 * boxes->data[asg_to_ind3(a3, boxes->shp)]);
			a3.dim3 = 1; yi = floorf(x->shp.dim3 * boxes->data[asg_to_ind3(a3, boxes->shp)]);
			a3.dim3 = 2; xf = ceilf(x->shp.dim4 * boxes->data[asg_to_ind3(a3, boxes->shp)]);
			a3.dim3 = 3; yf = ceilf(x->shp.dim3 * boxes->data[asg_to_ind3(a3, boxes->shp)]);
			
			for(i=0; i<grad->shp.dim3; i++)
			{
				for(j=0; j<grad->shp.dim4; j++)
				{
					// define column/rows to loop over
					r_min = yi + floorf((yf - yi) * float(i) / grad->shp.dim3);
					c_min = xi + floorf((xf - xi) * float(j) / grad->shp.dim4);
					r_max = yi + min(ceilf((yf - yi) * float(i + 1) / grad->shp.dim3), yf - yi);
					c_max = xi + min(ceilf((xf - xi) * float(j + 1) / grad->shp.dim4), xf - xi);
					
					// find maximum element
					a4.dim1 = n; a4.dim2 = channel; a4.dim3 = r_min; a4.dim4 = c_min;
					max_ind = asg_to_ind4(a4, x->shp);
					max_val = x->data[max_ind];
					for(k=r_min; k<r_max; k++)
					{
						for(l=c_min; l<c_max; l++)
						{
							a4.dim3 = k; a4.dim4 = l;
							ind = asg_to_ind4(a4, x->shp);
							if(x->data[ind] > max_val)
							{
								max_val = x->data[ind];
								max_ind = ind;
							}
						}
					}
					
					a4.dim1 = n * boxes->shp.dim2 + m; a4.dim2 = channel; a4.dim3 = i; a4.dim4 = j;
					x_pool_grad->data[max_ind] += grad->data[asg_to_ind4(a4, grad->shp)];
				}
			}
		}
	}
"""

class Tensor3Struct:
	mem_size = 8 * 3 + np.intp(0).nbytes
	def __init__(self, array, ptr):
		assert(len(array.shape) == 3) 
		if isinstance(array, pygpu.GpuArray):
			self.data = array.gpudata
		else:
			if array.dtype != np.float32:
				array = array.astype(np.float32)
			self.data = cuda.to_device(array)
		self.shape = array.shape
		self.dtype = array.dtype
		cuda.memcpy_htod(int(ptr), np.getbuffer(np.int32(array.shape[0])))
		cuda.memcpy_htod(int(ptr) + 8, np.getbuffer(np.int32(array.shape[1])))
		cuda.memcpy_htod(int(ptr) + 16, np.getbuffer(np.int32(array.shape[2])))
		cuda.memcpy_htod(int(ptr) + 24, np.getbuffer(np.intp(int(self.data))))
	
	def get_val(self):
		return cuda.from_device(self.data, self.shape, self.dtype)
	
class Tensor4Struct:
	mem_size = 8 * 4 + np.intp(0).nbytes
	def __init__(self, array, ptr):
		assert(len(array.shape) == 4)
		if isinstance(array, pygpu.GpuArray):
			self.data = array.gpudata
		else:
			if array.dtype != np.float32:
				array = array.astype(np.float32)
			self.data = cuda.to_device(array)
		self.shape = array.shape
		self.dtype = array.dtype
		cuda.memcpy_htod(int(ptr), np.getbuffer(np.int32(array.shape[0])))
		cuda.memcpy_htod(int(ptr) + 8, np.getbuffer(np.int32(array.shape[1])))
		cuda.memcpy_htod(int(ptr) + 16, np.getbuffer(np.int32(array.shape[2])))
		cuda.memcpy_htod(int(ptr) + 24, np.getbuffer(np.int32(array.shape[3])))
		cuda.memcpy_htod(int(ptr) + 32, np.getbuffer(np.intp(int(self.data))))
	
	def get_val(self):
		return cuda.from_device(self.data, self.shape, self.dtype)
		
def get_tens_ptr(array):
	if array.ndim == 3:
		ptr = cuda.mem_alloc(Tensor3Struct.mem_size)
		tens = Tensor3Struct(array, ptr)
		return ptr, tens
	elif array.ndim == 4:
		ptr = cuda.mem_alloc(Tensor4Struct.mem_size)
		tens = Tensor4Struct(array, ptr)
		return ptr, tens
	else:
		raise Exception("invalid number of dims")

class PyCUDAROIPool(theano.Op):
	__props__ = ()
	
	def __init__(self, shape):
		self.shape = shape
	
	def make_node(self, x, boxes):
		x = basic_ops.gpu_contiguous(x)
		boxes = basic_ops.gpu_contiguous(boxes)

		return theano.Apply(self, [x,boxes], [x.type()])
	
	def infer_shape(self, node, ishapes):
		xshape, bshape = ishapes[0], ishapes[1]
		oshape = tuple([
			bshape[0] * bshape[1],
			xshape[1],
			self.shape[0],
			self.shape[1]
		])
		return [oshape]
	
	def grad(self, inputs, grads):
		x, boxes = inputs[0], inputs[1]
		boxes_grad = theano.gradient.grad_undefined(self, 1, boxes, "PyCUDAROIPool doesn't have a gradient w.r.t. its parameters.")
		x_grad = PyCUDAROIPoolGrad(self.shape)(x, boxes, grads[0])
		return [x_grad, boxes_grad]
	
	def make_thunk(self, node, storage_map, _, _2, impl=None):
#		 mod = SourceModule(roi_code)
		roi_mod = SourceModule(roi_code)
		pycuda_func = roi_mod.get_function("roi_pool")
		inputs = [storage_map[v] for v in node.inputs]
		outputs = [storage_map[v] for v in node.outputs]
		
		def thunk():
			x, boxes = inputs[0], inputs[1]
			context = None
			if hasattr(x[0], 'context'):
				context = x[0].context
			z = outputs[0]
			z_shape = (
				np.prod(boxes[0].shape[:2]),
				x[0].shape[1],
				self.shape[0],
				self.shape[1]
			)
			if z[0] is None or z[0].shape != z_shape:
				z[0] = pygpu.zeros(z_shape, dtype=theano.config.floatX, context=context)
			x_ptr, _ = get_tens_ptr(x[0])
			boxes_ptr, _ = get_tens_ptr(boxes[0])
			z_ptr, z_tens = get_tens_ptr(z[0])
			grid = (self.shape[0], self.shape[1], z_shape[0])
			block = (x[0].shape[1], 1, 1)
			pycuda_func(z_ptr, x_ptr, boxes_ptr, block=block, grid=grid)
		
		return thunk

class PyCUDAROIPoolGrad(theano.Op):
	__props__ = ()
	
	def __init__(self, shape):
		self.shape = shape
	
	def make_node(self, x, boxes, grad):
		x = basic_ops.gpu_contiguous(x)
		boxes = basic_ops.gpu_contiguous(boxes)
		grad = basic_ops.gpu_contiguous(grad)

		return theano.Apply(self, [x,boxes,grad], [x.type()])
	
	def infer_shape(self, node, ishapes):
		xshape, bshape = ishapes[0], ishapes[1]
		return [xshape]
	
	def make_thunk(self, node, storage_map, _, _2, impl=None):
#		 mod = SourceModule("roi")
		roi_mod = SourceModule(roi_code)
		pycuda_func = roi_mod.get_function("roi_pool_grad")
		inputs = [storage_map[v] for v in node.inputs]
		outputs = [storage_map[v] for v in node.outputs]
		
		def thunk():
			x, boxes, grad = inputs[0], inputs[1], inputs[2]
			context = None
			if hasattr(x[0], 'context'):
				context = x[0].context
			z = outputs[0]
			if z[0] is None or z[0].shape != x[0].shape:
				z[0] = pygpu.zeros(x[0].shape, dtype=theano.config.floatX, context=context)
			else:
				z[0][:] = 0
			x_ptr, _ = get_tens_ptr(x[0])
			boxes_ptr, _ = get_tens_ptr(boxes[0])
			grad_ptr, _ = get_tens_ptr(grad[0])
			z_ptr, z_tens = get_tens_ptr(z[0])
			grid = (x[0].shape[0], x[0].shape[1], 1)
			block = (1,1,1)
			pycuda_func(z_ptr, x_ptr, boxes_ptr, grad_ptr, block=block, grid=grid)
		return thunk
	
def roi_pool(x, boxes, shape):
	'''
	Theano operation for ROI pooling -- uses custom PyCUDA code under the hood.

	Parameters
	----------
	x : theano.tensor.tensor4
		Input feature map -- `x`.shape = (number of images, number of channels, height, width).
	boxes : theano.tensor.tensor3
		Input boxes -- `boxes`.shape = (number of images, number of regions, 4)
	shape : tuple
		What size do you want to reshape the input

	Returns
	-------
	theano.tensor.tensor4
		ROI pooled feature map with shape (number of image :math:`\\times` number of regions, number of channels, shape[0], shape[1]).
	'''
	return PyCUDAROIPool(shape)(x, boxes)


class ROILayer(Layer):
	'''
	Lasagne layer for the ROI pooling layer.

	Parameters
	----------
	incoming : lasagne.layers.BaseLayer
		Incoming layer.
	shape : tuple
		Shape of ROI pooling.
	boxes : theano.tensor.tensor3, or None (default None)
		Theano symbolic variable for the boxes
	'''

	def __init__(self, incoming, shape, boxes=None, **kwargs):
		super(ROILayer, self).__init__(incoming, **kwargs)
		if boxes is None:
			boxes = T.tensor3('roi_boxes')
		self.boxes = boxes
		self.shape = shape

	def get_output_for(self, input, **kwargs):
		return roi_pool(input, self.boxes, self.shape)

	def get_output_shape_for(self, input_shape):
		n_rois = input_shape[0] if input_shape[0] is None else inpuse_shape[0] * self.boxes.shape[1]
		return (n_rois, input_shape[1], self.shape[0], self.shape[1])




