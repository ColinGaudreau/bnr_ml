import theano
# import theano.sandbox.cuda as tcuda
import theano.misc.pycuda_init
import theano.tensor as T
import theano.gpuarray as pygpu

from lasagne.layers import Layer

import numpy as np
import numpy.random as npr

import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

import pdb

yolo_code = """
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
	
	struct yolo_info {
		int n_classes, __pd1;
		int n_anchors, __pd2;
		float l_obj, __pd3;
		float l_noobj, __pd4;
		float *anchors;
	};
	
	__device__ asg4 ind_to_asg4(int idx, shape4 shp)
	{
		asg4 asg;
		asg.dim1 = floorf(idx / (shp.dim2 * shp.dim3 * shp.dim4));
		asg.dim2 = ((int) floorf(idx / (shp.dim3 * shp.dim4))) % shp.dim2;
		asg.dim3 = ((int) floorf(idx / shp.dim4)) % shp.dim3;
		asg.dim4 = idx % shp.dim4;
		return asg;
	}
	
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
	
	__device__ asg3 make_asg3(int dim1, int dim2, int dim3)
	{
		asg3 asg;
		asg.dim1 = dim1; asg.dim2 = dim2; asg.dim3 = dim3;
		return asg;
	}
	
	__device__ asg4 make_asg4(int dim1, int dim2, int dim3, int dim4)
	{
		asg4 asg;
		asg.dim1 = dim1; asg.dim2 = dim2; asg.dim3 = dim3; asg.dim4 = dim4;
		return asg;
	}
	
	__device__ float iou_box(float *box1, float *box2)
	{
		float xi, yi, xf, yf;
		float un, isec, iou, w, h;
		// find bounds of intersection
		xi = max(box1[0], box2[0]); yi = max(box1[1], box2[1]);
		xf = min(box1[0]+box1[2], box2[0]+box2[2]); yf = min(box1[1]+box1[3], box2[1]+box2[3]);
		w = max(0., xf - xi); h = max(0., yf - yi);
		isec = w * h;
		un = box1[2]*box1[3] + box2[2]*box2[3] - isec;
		iou = 0;
		if(un > 0)
			iou = isec / un;
		return iou;
	}
	
	__device__ float dist_box(float *box1, float *box2)
	{
		return sqrtf(powf(box1[0]-box2[0],2) + powf(box1[1]-box2[1],2)
			+ powf(box1[0]+box1[2]-box2[0]-box2[2],2) + powf(box1[1]+box1[3]-box2[1]-box2[3],2));
	}
	
	__device__ void get_pred_box(float *box, tens4 *predictions, int N, int anchor, int row, int col, yolo_info *info)
	{
		box[0] = predictions->data[asg_to_ind4(make_asg4(N,0+anchor*(5+info->n_classes),row,col),predictions->shp)];
		box[1] = predictions->data[asg_to_ind4(make_asg4(N,1+anchor*(5+info->n_classes),row,col),predictions->shp)];
		box[2] = info->anchors[2*anchor] * expf(predictions->data[asg_to_ind4(make_asg4(N,2+anchor*(5+info->n_classes),row,col),predictions->shp)]);
		box[3] = info->anchors[2*anchor+1] * expf(predictions->data[asg_to_ind4(make_asg4(N,3+anchor*(5+info->n_classes),row,col),predictions->shp)]);
		
		// network predicts the center of the bounding box, we need to set x_1, x_2 to be the top left corner
		box[0] += ((col+0.5)/predictions->shp.dim4 - box[2]/2);
		box[1] += ((row+0.5)/predictions->shp.dim3 - box[3]/2);
	}
	
	__device__ void get_truth_box(float *box, tens3 *truth, int N, int gt)
	{
		box[0] = truth->data[asg_to_ind3(make_asg3(N,gt,0),truth->shp)];
		box[1] = truth->data[asg_to_ind3(make_asg3(N,gt,1),truth->shp)];
		box[2] = truth->data[asg_to_ind3(make_asg3(N,gt,2),truth->shp)];
		box[3] = truth->data[asg_to_ind3(make_asg3(N,gt,3),truth->shp)];
	}
	
	__device__ float l2_cost_fn(float val)
	{
		return powf(val, 2);
	}
	
	__device__ float l2_grad_fn(float val)
	{
		return 2 * val;
	}
	
	__global__ void assign_boxes(int *best_indices, float *best_ious, tens4 *predictions, tens3 *truth, yolo_info *info)
	{
		/*
			This function spreads only the examples across different blocks.
		*/
		int N, i, j, k, l, m, new_idx, best_idx;
		float best_iou, new_iou;
		float truth_box[4];
		float pred_box[4];
		bool found;
		float MIN_IOU = .01;
		N = blockIdx.x;
		
		// loop over every bounding box to assign each truth to a bounding box
		for(i=0; i<truth->shp.dim2; i++)
		{
			best_idx = -1; best_iou = -1;
			get_truth_box(truth_box, truth, N, i); // fills box with gt boxes
			
			// loop over grid of predictions
			for(j=0; j<predictions->shp.dim3; j++)
			{
				for(k=0; k<predictions->shp.dim4; k++)
				{
					for(l=0; l<info->n_anchors; l++)
					{
						get_pred_box(pred_box, predictions, N, l, j, k, info); // fills box with predictions
						
						new_iou = iou_box(pred_box, truth_box);
						new_idx = asg_to_ind4(make_asg4(N,l*(5+info->n_classes),j,k), predictions->shp);
						if(new_iou > best_iou)
						{
							found = false;
							// check that bounding box hasn't already been assigned
							for(m=0; m<i; m++)
							{
								if(best_indices[truth->shp.dim1 + m] == new_idx)
									found = true;
							}
							if(!found)
							{
								best_iou = new_iou; best_idx = new_idx;
							}
						}
					}
				}
			}
			
			// non-object ground truths are set to some box that doesn't overlap the frame
			if(best_iou > MIN_IOU)
			{
				best_ious[N*truth->shp.dim2 + i] = best_iou;
				best_indices[N*truth->shp.dim2 + i] = best_idx;
			} else {
				best_ious[N*truth->shp.dim2 + i] = -1;
				best_indices[N*truth->shp.dim2 + i] = -1;
			}
		}
	}
	
	__global__ void yolo_v2_cost(tens4 *cost, int *best_indices, float *best_ious, tens4 *predictions, tens3 *truth, yolo_info *info)
	{
		// define cost function
		float (*cost_fn)(float);
		cost_fn = &l2_cost_fn;
		
		int N, s1, s2, anchor, idx_pred, idx_truth, match_idx, i;
		float val, truth_val, mult_fact;
		asg4 asg;
		bool chosen = false;
		N = blockIdx.x; s1 = blockIdx.y; s2 = blockIdx.z;
		anchor = threadIdx.x;
		mult_fact = 1. / predictions->shp.dim1; // takes the mean
		
		// check if box was matched
		for(match_idx=0; match_idx<truth->shp.dim2; match_idx++)
		{
			asg = ind_to_asg4(best_indices[N*truth->shp.dim2+match_idx], cost->shp);
			if(asg.dim1==N && asg.dim2==anchor*(5+info->n_classes) && asg.dim3==s1 && asg.dim4==s2)
			{
				chosen = true;
				break;
			}
		}
		
		// cost for x coordinate
		idx_pred = asg_to_ind4(make_asg4(N, 0+anchor*(5+info->n_classes),s1,s2), cost->shp);
		val = predictions->data[idx_pred];
		if(chosen)
		{
			val = val + (0.5 + s2) / predictions->shp.dim4;
			truth_val = truth->data[asg_to_ind3(make_asg3(N, match_idx, 0), truth->shp)] + truth->data[asg_to_ind3(make_asg3(N, match_idx, 2), truth->shp)]/2;
			cost->data[idx_pred] = mult_fact * info->l_obj * cost_fn(val - truth_val);
		} else {
			cost->data[idx_pred] = mult_fact * info->l_noobj * cost_fn(val); // regress to anchor
		}
		
		// cost for y coordinate
		idx_pred = asg_to_ind4(make_asg4(N, 1+anchor*(5+info->n_classes),s1,s2), cost->shp);
		val = predictions->data[idx_pred];
		if(chosen)
		{
			val = val + (0.5 + s1) / predictions->shp.dim3;
			truth_val = truth->data[asg_to_ind3(make_asg3(N, match_idx, 1), truth->shp)] + truth->data[asg_to_ind3(make_asg3(N, match_idx, 3), truth->shp)]/2;
			cost->data[idx_pred] = mult_fact * info->l_obj * cost_fn(val - truth_val);
		} else {
			cost->data[idx_pred] = mult_fact * info->l_noobj * cost_fn(val); // regress to anchor
		}
		
		// cost for w
		idx_pred = asg_to_ind4(make_asg4(N, 2+anchor*(5+info->n_classes),s1,s2), cost->shp);
		val = predictions->data[idx_pred];
		if(chosen)
		{
			idx_truth = asg_to_ind3(make_asg3(N, match_idx, 2), truth->shp);
			cost->data[idx_pred] = mult_fact * info->l_obj * cost_fn(val - logf(truth->data[idx_truth] / info->anchors[2*anchor]));
		} else {
			cost->data[idx_pred] = mult_fact * info->l_noobj * cost_fn(val); // regress to anchor
		}
		
		// cost for h
		idx_pred = asg_to_ind4(make_asg4(N, 3+anchor*(5+info->n_classes),s1,s2), cost->shp);
		val = predictions->data[idx_pred];
		if(chosen)
		{
			idx_truth = asg_to_ind3(make_asg3(N, match_idx, 3), truth->shp);
			cost->data[idx_pred] = mult_fact * info->l_obj * cost_fn(val - logf(truth->data[idx_truth] / info->anchors[2*anchor+1]));
		} else {
			cost->data[idx_pred] = mult_fact * info->l_noobj * cost_fn(val); // regress to anchor
		}
		
		// cost for objectness
		idx_pred = asg_to_ind4(make_asg4(N, 4+anchor*(5+info->n_classes),s1,s2), cost->shp);
		val = predictions->data[idx_pred];
		if(chosen)
		{
			cost->data[idx_pred] = mult_fact * info->l_obj * cost_fn(val - best_ious[N*truth->shp.dim2+match_idx]);
		} else {
			cost->data[idx_pred] = mult_fact * info->l_noobj * cost_fn(val); // regress to anchor
		}
		
		if(chosen)
		{
			for(i=0; i<info->n_classes; i++)
			{
				idx_pred = asg_to_ind4(make_asg4(N, i+5+anchor*(5+info->n_classes),s1,s2), cost->shp);
				idx_truth = asg_to_ind3(make_asg3(N, match_idx, 4+i), truth->shp);
				cost->data[idx_pred] = -mult_fact * info->l_obj * truth->data[idx_truth] * logf(predictions->data[idx_pred]); // log loss
			}
		}
	}
	
	__global__ void yolo_v2_grad(tens4 *grad, int *best_indices, float *best_ious, tens4 *predictions, tens3 *truth, yolo_info *info)
	{
		// define grad function
		float (*grad_fn)(float);
		grad_fn = &l2_grad_fn;
		
		int N, s1, s2, anchor, idx_pred, idx_truth, match_idx, i;
		float val, truth_val, mult_fact;
		bool chosen = false;
		N = blockIdx.x; s1 = blockIdx.y; s2 = blockIdx.z;
		anchor = threadIdx.x;
		asg4 asg;
		mult_fact = 1./predictions->shp.dim1;
		
		// check if box was matched
		for(match_idx=0; match_idx<truth->shp.dim2; match_idx++)
		{
			asg = ind_to_asg4(best_indices[N*truth->shp.dim2+match_idx], grad->shp);
			if(asg.dim1==N && asg.dim2==anchor*(5+info->n_classes) && asg.dim3==s1 && asg.dim4==s2)
			{
				chosen = true;
				break;
			}
		}
		
		// grad for x coordinate
		idx_pred = asg_to_ind4(make_asg4(N, 0+anchor*(5+info->n_classes),s1,s2), grad->shp);
		val = predictions->data[idx_pred];
		if(chosen)
		{
			val = val + (0.5 + s2) / predictions->shp.dim4;
			truth_val = truth->data[asg_to_ind3(make_asg3(N, match_idx, 0), truth->shp)] + truth->data[asg_to_ind3(make_asg3(N, match_idx, 2), truth->shp)]/2;
			grad->data[idx_pred] = mult_fact*info->l_obj * grad_fn(val - truth_val);
		} else {
			grad->data[idx_pred] = mult_fact * info->l_noobj * grad_fn(val); // regress to anchor
		}
		
		// grad for y coordinate
		idx_pred = asg_to_ind4(make_asg4(N, 1+anchor*(5+info->n_classes),s1,s2), grad->shp);
		val = predictions->data[idx_pred];
		if(chosen)
		{
			val = val + (0.5 + s1) / predictions->shp.dim3;
			truth_val = truth->data[asg_to_ind3(make_asg3(N, match_idx, 1), truth->shp)] + truth->data[asg_to_ind3(make_asg3(N, match_idx, 3), truth->shp)]/2;
			grad->data[idx_pred] = mult_fact * info->l_obj * grad_fn(val - truth_val);
		} else {
			grad->data[idx_pred] = mult_fact * info->l_noobj * grad_fn(val); // regress to anchor
		}
		
		// grad for w
		idx_pred = asg_to_ind4(make_asg4(N, 2+anchor*(5+info->n_classes),s1,s2), grad->shp);
		val = predictions->data[idx_pred];
		if(chosen)
		{
			idx_truth = asg_to_ind3(make_asg3(N, match_idx, 2), truth->shp);
			grad->data[idx_pred] = mult_fact * info->l_obj * grad_fn(val - logf(truth->data[idx_truth] / info->anchors[2*anchor]));
		} else {
			grad->data[idx_pred] = mult_fact * info->l_noobj * grad_fn(val); // regress to anchor
		}
		
		// grad for h
		idx_pred = asg_to_ind4(make_asg4(N, 3+anchor*(5+info->n_classes),s1,s2), grad->shp);
		val = predictions->data[idx_pred];
		if(chosen)
		{
			idx_truth = asg_to_ind3(make_asg3(N, match_idx, 3), truth->shp);
			grad->data[idx_pred] = mult_fact * info->l_obj * grad_fn(val - logf(truth->data[idx_truth] / info->anchors[2*anchor+1]));
		} else {
			grad->data[idx_pred] = mult_fact * info->l_noobj * grad_fn(val); // regress to anchor
		}
		
		// grad for objectness
		idx_pred = asg_to_ind4(make_asg4(N, 4+anchor*(5+info->n_classes),s1,s2), grad->shp);
		val = predictions->data[idx_pred];
		if(chosen)
		{
			grad->data[idx_pred] = mult_fact * info->l_obj * grad_fn(val - best_ious[N*truth->shp.dim2+match_idx]);
		} else {
			grad->data[idx_pred] = mult_fact * info->l_noobj * grad_fn(val); // regress to anchor
		}
		
		if(chosen)
		{
			for(i=0; i<info->n_classes; i++)
			{
				idx_pred = asg_to_ind4(make_asg4(N, i+5+anchor*(5+info->n_classes),s1,s2), grad->shp);
				idx_truth = asg_to_ind3(make_asg3(N, match_idx, 4+i), truth->shp);
				grad->data[idx_pred] = -(mult_fact * info->l_obj * truth->data[idx_truth]) / predictions->data[idx_pred]; // log loss
			}
		}
	}
"""

class YoloInfo:
	mem_size = 8*4 + np.intp(0).nbytes
	def __init__(self, n_classes, n_anchors, l_obj, l_noobj, anchors, ptr):
		array = np.asarray(anchors, dtype=np.float32)
		self.anchors = cuda.to_device(array)
		cuda.memcpy_htod(int(ptr), np.getbuffer(np.int32(n_classes)))
		cuda.memcpy_htod(int(ptr)+8, np.getbuffer(np.int32(n_anchors)))
		cuda.memcpy_htod(int(ptr)+16, np.getbuffer(np.float32(l_obj)))
		cuda.memcpy_htod(int(ptr)+24, np.getbuffer(np.float32(l_noobj)))
		cuda.memcpy_htod(int(ptr)+32, np.getbuffer(np.intp(int(self.anchors))))

def get_yolo_info(n_classes, n_anchors, l_obj, l_noobj, anchors):
	new_anchors = []
	for anchor in anchors:
		new_anchors.extend(anchor)
	ptr = cuda.mem_alloc(YoloInfo.mem_size)
	return ptr, YoloInfo(n_classes, n_anchors, l_obj, l_noobj, new_anchors, ptr)

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

class PyCUDAYolo2Cost(theano.Op):
	__props__ = ()
	def __init__(self, n_classes, n_anchors, l_obj, l_noobj, anchors):
		self.n_classes = n_classes
		self.n_anchors = n_anchors
		self.l_obj = l_obj
		self.l_noobj = l_noobj
		self.anchors = anchors
	
	def make_node(self, x, truth):
		# x = tcuda.basic_ops.gpu_contiguous(
		# 	tcuda.basic_ops.as_cuda_ndarray_variable(x)
		# )
		# truth = tcuda.basic_ops.gpu_contiguous(
		# 	tcuda.basic_ops.as_cuda_ndarray_variable(truth)
		# )
		return theano.Apply(self, [x,truth], [T.scalar()])
	
	def infer_shape(self, node, ishapes):
		return [ishapes[0][:0]]
	
	def grad(self, inputs, grads):
		x, truth = inputs[0], inputs[1]
		truth_grad = theano.gradient.grad_undefined(self, 1, truth, "PyCUDAYolo2Cost doesn't have a gradient w.r.t. the labels.")
		x_grad = PyCUDAYolo2CostGrad(self.n_classes, self.n_anchors, self.l_obj, self.l_noobj, self.anchors)(x, truth)
		return [x_grad, truth_grad]

	def make_thunk(self, node, storage_map, _, _2, impl=None):
		yolo_mod = SourceModule(yolo_code)
		index_fn = yolo_mod.get_function("assign_boxes")
		cost_fn = yolo_mod.get_function("yolo_v2_cost")
		inputs = [storage_map[v] for v in node.inputs]
		outputs = [storage_map[v] for v in node.outputs]
		n_classes, n_anchors, l_obj, l_noobj, anchors = self.n_classes, self.n_anchors, self.l_obj, self.l_noobj, self.anchors
		
		def thunk():
			x, truth = inputs[0], inputs[1]
			z = outputs[0]
			z_shape = (
				x[0].shape[:0],
			)
			if z[0] is None or z[0].shape != z_shape:
				z[0] = None
			x_ptr, _ = get_tens_ptr(x[0])
			truth_ptr, _ = get_tens_ptr(truth[0])
			cost_ptr, cost_obj = get_tens_ptr(np.zeros_like(x[0], dtype=theano.config.floatX))
			best_idx_ptr = cuda.mem_alloc(8 * truth[0].shape[1] * truth[0].shape[0])
			best_iou_ptr = cuda.mem_alloc(8 * truth[0].shape[1] * truth[0].shape[0])
			yolo_ptr, _ = get_yolo_info(n_classes, n_anchors, l_obj, l_noobj, anchors)

			# get best index
			index_fn(best_idx_ptr, best_iou_ptr, x_ptr, truth_ptr, yolo_ptr, block=(1,1,1), grid=(x[0].shape[0],1,1))
			cost_fn(cost_ptr, best_idx_ptr, best_iou_ptr, x_ptr, truth_ptr, yolo_ptr, block=(n_anchors,1,1), grid=(x[0].shape[0],x[0].shape[2],x[0].shape[3]))
			tmp = gpuarray.sum(gpuarray.GPUArray(cost_obj.shape, cost_obj.dtype, gpudata=cost_obj.data))
			foo = np.zeros(1, dtype=np.float32)
			tmp.get(foo)
			z[0] = foo[0]

			# free all memory
			x_ptr.free(); truth_ptr.free(); cost_ptr.free(); best_idx_ptr.free(); best_iou_ptr.free(); yolo_ptr.free()

		return thunk

class PyCUDAYolo2CostGrad(theano.Op):
	__props__ = ()
	
	def __init__(self, n_classes, n_anchors, l_obj, l_noobj, anchors):
		self.n_classes = n_classes
		self.n_anchors = n_anchors
		self.l_obj = l_obj
		self.l_noobj = l_noobj
		self.anchors = anchors
	
	def make_node(self, x, truth):
		# x = tcuda.basic_ops.gpu_contiguous(
		# 	tcuda.basic_ops.as_cuda_ndarray_variable(x)
		# )
		# truth = tcuda.basic_ops.gpu_contiguous(
		# 	tcuda.basic_ops.as_cuda_ndarray_variable(truth)
		# )
		return theano.Apply(self, [x,truth], [x.type()])
	
	def infer_shape(self, node, ishapes):
		return [ishapes[0]]

	def make_thunk(self, node, storage_map, _, _2, impl=None):
		yolo_mod = SourceModule(yolo_code)
		index_fn = yolo_mod.get_function("assign_boxes")
		grad_fn = yolo_mod.get_function("yolo_v2_grad")
		inputs = [storage_map[v] for v in node.inputs]
		outputs = [storage_map[v] for v in node.outputs]
		n_classes, n_anchors, l_obj, l_noobj, anchors = self.n_classes, self.n_anchors, self.l_obj, self.l_noobj, self.anchors
		
		def thunk():
			x, truth = inputs[0], inputs[1]
			z = outputs[0]
			z_shape = x[0].shape
			if z[0] is None or z[0].shape != z_shape:
				z[0] = pygpu.zeros(z_shape)
			x_ptr, _ = get_tens_ptr(x[0])
			truth_ptr, _ = get_tens_ptr(truth[0])
			z_ptr, _ = get_tens_ptr(z[0])
			best_idx_ptr = cuda.mem_alloc(8 * truth[0].shape[1] * truth[0].shape[0])
			best_iou_ptr = cuda.mem_alloc(8 * truth[0].shape[1] * truth[0].shape[0])
			yolo_ptr, _ = get_yolo_info(n_classes, n_anchors, l_obj, l_noobj, anchors)

			# get best index
			index_fn(best_idx_ptr, best_iou_ptr, x_ptr, truth_ptr, yolo_ptr, block=(1,1,1), grid=(x[0].shape[0],1,1))
			grad_fn(z_ptr, best_idx_ptr, best_iou_ptr, x_ptr, truth_ptr, yolo_ptr, block=(n_anchors,1,1), grid=(x[0].shape[0],x[0].shape[2],x[0].shape[3]))

			# free all memory
			x_ptr.free(); truth_ptr.free(); z_ptr.free(); best_idx_ptr.free(); best_iou_ptr.free(); yolo_ptr.free()

		return thunk


def yolo2_cost(x, truth, n_classes, n_anchors, l_obj, l_noobj, anchors):
	return T.sum(PyCUDAYolo2Cost(n_classes, n_anchors, l_obj, l_noobj, anchors)(x, truth))

	
