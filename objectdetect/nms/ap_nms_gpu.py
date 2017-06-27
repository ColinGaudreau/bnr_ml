import numpy as np
from ap_nms import *
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

_cuda_code = '''
#include <stdio.h>

struct Shape{
    int dim1, __pd1;
    int dim2, __pd2;
};

struct Matrix{
    Shape shape;
    float *data;
};

__device__ void set_alpha(Matrix *alpha, Matrix *rho, int row, int col, float damping)
{
    float val;
    if(row == col)
    {
        val = 0;
        for(int i=0; i<rho->shape.dim1; i++)
        {
            if(i != col)
            {
                val += max(0., rho->data[col + i * rho->shape.dim2]);
            }
        }
    } else {
        val = rho->data[col + col * rho->shape.dim2];
        for(int i=0; i<rho->shape.dim2; i++)
        {
            if(i != row && i != col)
            {
                val += max(0., rho->data[col + i * rho->shape.dim2]);
            }
        }
        val = min(0., val);
    }
    alpha->data[col + row * alpha->shape.dim2] = damping * alpha->data[col + row * alpha->shape.dim2] + (1 - damping) * val;
}

__device__ void set_rho(Matrix *rho, Matrix *s_hat, Matrix *alpha, Matrix *phi, int row, int col, float damping)
{
    float val, max_val;
    bool chosen;
    
    if(row == col)
    {
        val = s_hat->data[row + row * rho->shape.dim2];
        chosen = false;
        max_val = 0;
        for(int i=0; i<rho->shape.dim2; i++)
        {
            if(i != col)
            {
                if(s_hat->data[i + row * rho->shape.dim2] + alpha->data[i + row * alpha->shape.dim2] > max_val || !chosen)
                {
                    max_val = s_hat->data[i + row * rho->shape.dim2] + alpha->data[i + row * alpha->shape.dim2];
                    chosen = true;
                }
                
                val += phi->data[i + row * rho->shape.dim2];
            }
        }
        val -= max_val;
    } else {
        val = s_hat->data[row + row * s_hat->shape.dim2] + alpha->data[row + row * alpha->shape.dim2];
        max_val = 0;
        chosen = false;
        for(int i=0; i<rho->shape.dim2; i++)
        {
            if(i != col)
            {
                if(i != row && (s_hat->data[i + row * rho->shape.dim2] + alpha->data[i + row * alpha->shape.dim2] > max_val || !chosen))
                {
                    max_val = s_hat->data[i + row * rho->shape.dim2] + alpha->data[i + row * alpha->shape.dim2];
                    chosen = true;
                }
                
                val += phi->data[i + row * rho->shape.dim2];
            }
        }
        val = max(max_val, val);
        val = s_hat->data[col + row * s_hat->shape.dim2] - val;
    }
    rho->data[col + row * rho->shape.dim2] = damping * rho->data[col + row * rho->shape.dim2] + (1 - damping) * val;
}

__device__ void set_gamma(Matrix *gamma, Matrix *s_hat, Matrix *alpha, Matrix *phi, int row, int col, float damping)
{
    float val, max_val;
    bool chosen;
    
    val = s_hat->data[row + row * s_hat->shape.dim2] + alpha->data[row + row * s_hat->shape.dim2];
    max_val = 0;
    chosen = false;
    for(int i=0; i<gamma->shape.dim2; i++)
    {
        if(i != row)
        {
            if(!chosen || s_hat->data[i + row * s_hat->shape.dim2] + alpha->data[i + row * alpha->shape.dim2] > max_val)
            {
                max_val = s_hat->data[i + row * s_hat->shape.dim2] + alpha->data[i + row * alpha->shape.dim2];
                chosen = true;
            }
            
            if(i != col)
            {
                val += phi->data[i + row * phi->shape.dim2];
            }
        }
    }
    val -= max_val;
    if(row == col)
        val = 0;
    gamma->data[col + row * gamma->shape.dim2] = damping * gamma->data[col + row * gamma->shape.dim2] + (1 - damping) * val;
}

__device__ void set_phi(Matrix *phi, Matrix *gamma, Matrix *r_hat, int row, int col, float damping)
{
    float val1 = max(0., gamma->data[row + col * gamma->shape.dim2] + r_hat->data[col + row * r_hat->shape.dim2]); 
    float val2 = max(0., gamma->data[row + col * gamma->shape.dim2]);
    float val = val1 - val2;
    if(isnan(val) || row == col)
        val = 0;
    phi->data[col + row * phi->shape.dim2] = damping * phi->data[col + row * phi->shape.dim2] + (1 - damping) * val;
}

__global__ void affinity_propagation(Matrix *s_hat, Matrix *r_hat, Matrix *rho, Matrix *alpha, Matrix *gamma, Matrix *phi, int iterations, float damping)
{
    int row, col, i;
    row = blockIdx.x;
    col = threadIdx.x;
    
    for(i=0; i<iterations; i++)
    {
        set_alpha(alpha, rho, row, col, damping);
        set_phi(phi, gamma, r_hat, row, col, damping);
        __syncthreads();
        set_rho(rho, s_hat, alpha, phi, row, col, damping);
        set_gamma(gamma, s_hat, alpha, phi, row, col, damping);
        __syncthreads();
    }
}
'''

class MatrixStruct:
    mem_size = 8 * 2 + np.intp(0).nbytes
    def __init__(self, array, ptr):
        assert(len(array.shape) == 2)
        if isinstance(array, gpuarray.GPUArray):
            self.data = array.gpudata
        else:
            if array.dtype != np.float32:
                array = array.astype(np.float32)
            self.data = cuda.to_device(array)
        self.shape = array.shape
        self.dtype = array.dtype
        cuda.memcpy_htod(int(ptr), np.getbuffer(np.int32(array.shape[0])))
        cuda.memcpy_htod(int(ptr) + 8, np.getbuffer(np.int32(array.shape[1])))
        cuda.memcpy_htod(int(ptr) + 16, np.getbuffer(np.intp(int(self.data))))
    
    def get_val(self):
        return cuda.from_device(self.data, self.shape, self.dtype)
        
def get_ptr(array):
    ptr = cuda.mem_alloc(MatrixStruct.mem_size)
    mat = MatrixStruct(array, ptr)
    return ptr, mat

_module = SourceModule(_cuda_code)

_ap_func = _module.get_function('affinity_propagation')

def affinity_propagation_gpu(S, iterations=10, tol=1e-5, damping=0.5, print_every=2, w=[1.,1.,1.,1.]):
    wa, wb, wc, wd = w[0], w[1], w[2], w[3]
    s_hat = get_s_hat(S, wa, wb, wc).astype(np.float32)
    c = np.eye(S.shape[0])
    R = -(S + 1.) * wd
    
    diag_idx = np.arange(R.shape[0])
    R[diag_idx, diag_idx] = 0.

    r_hat = get_r_hat(R).astype(np.float32)
    alpha = np.zeros_like(s_hat)
    gamma = np.zeros_like(s_hat)
    rho = np.zeros_like(s_hat)
    phi = np.zeros_like(s_hat)

    ll_old = 0;

    s_hat_d, r_hat_d, alpha_d = gpuarray.to_gpu(s_hat), gpuarray.to_gpu(r_hat), gpuarray.to_gpu(alpha)
    gamma_d, rho_d, phi_d = gpuarray.to_gpu(gamma), gpuarray.to_gpu(rho), gpuarray.to_gpu(phi)
    grid = (s_hat.shape[0], 1, 1)
    block = (s_hat.shape[1], 1, 1)

    s_hat_ptr, _ = get_ptr(s_hat_d)
    r_hat_ptr, _ = get_ptr(r_hat_d)
    alpha_ptr, _ = get_ptr(alpha_d)
    gamma_ptr, _ = get_ptr(gamma_d)
    rho_ptr, _ = get_ptr(rho_d)
    phi_ptr, _ = get_ptr(phi_d)
    
    _ap_func(s_hat_ptr, r_hat_ptr, rho_ptr, alpha_ptr, gamma_ptr, phi_ptr, np.int32(iterations), np.float32(damping), block=block, grid=grid)
    #for itr in range(iterations):
     #   _ap_func(s_hat_ptr, r_hat_ptr, rho_ptr, alpha_ptr, gamma_ptr, phi_ptr, np.int32(1), np.float32(damping), block=block, grid=grid)

    # ll = gpuarray.sum(alpha_d + gamma_d + rho_d + phi_d).get()
    return get_labels(alpha_d.get(), phi_d.get(), rho_d.get(), gamma_d.get())










