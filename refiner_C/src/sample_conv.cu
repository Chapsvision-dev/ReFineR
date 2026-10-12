#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#include "helpers.h"


#define BS 32

namespace refiner_C{
    __global__ void sample_conv_kernel(
        float const *A,
        float *out,
        int const kernel_size, 
        int const stride,
        int C, int H, int W, int Hk, int Wk, int M
    ){
        int Mx = (threadIdx.z % M);
        int My = (threadIdx.z / M);

        int outx = (Mx + blockIdx.x * M);
        int outy = (My + blockIdx.y * M);
        if (outx >= Wk || outy >= Hk){return;}

        extern __shared__ float sdata[];

        int ds = kernel_size + stride * M;
        int sx = threadIdx.x + Mx * stride;
        int sy = threadIdx.y + My * stride;

        int ax = blockIdx.x * stride * M + sx;
        int ay = blockIdx.y * stride * M + sy;

        if (threadIdx.z == 0 || (sx >= kernel_size + (Mx - 1)*stride) || (sy >= kernel_size + (My - 1)*stride)){
            sdata[sx + sy * ds] = A[ax + ay * W + blockIdx.z * W * H];
        }

        int d0 = kernel_size;
        int d1 = kernel_size * d0;
        int d2 = Wk * d1;
        int d3 = Hk * d2;
        __syncthreads();
        out[threadIdx.x + threadIdx.y * d0 + outx * d1 + outy * d2 + blockIdx.z * d3] = sdata[sx + sy * ds];
    }
    void sample_conv_core(
        float const *A,
        float *out,
        int const kernel_size,
        int const stride,
        int C, int H, int W
    ){  
        int Hk = (H - kernel_size) / stride + 1;
        int Wk = (W - kernel_size) / stride + 1;
        int M = 16 / kernel_size; // M * M * kernel_size * kernel_size = 256 = 16 * 16

        dim3 dimGrid;
        dimGrid.x = (Wk + M - 1) / M;
        dimGrid.y = (Hk + M - 1) / M;
        dimGrid.z = C;

        dim3 dimBlock;
        dimBlock.x = kernel_size;
        dimBlock.y = kernel_size;
        dimBlock.z = M * M;

        sample_conv_kernel<<<dimGrid, dimBlock, ((kernel_size + M * stride) * (kernel_size + M * stride)) * sizeof(float)>>>(
            A, out, kernel_size, stride, C, H, W, Hk, Wk, M);
    }
    

    torch::Tensor sample_conv(
        const torch::Tensor A, // C x H x W
        int64_t const kernel_size,
        int64_t const stride
    ) {  
        // tensor sample with stride 1
        const int W = A.sizes()[2];
        const int H = A.sizes()[1];
        const int C = A.sizes()[0];


        auto out_options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
        torch::Tensor out = torch::empty({C, (H - kernel_size) / stride + 1, (W - kernel_size) / stride + 1, kernel_size, kernel_size}, out_options);

        torch::Tensor _A = A.contiguous();
        torch::Tensor _out = out.contiguous();
    
        float const *ptrA = reinterpret_cast<float*>(_A.data_ptr());
        float *ptrout = reinterpret_cast<float*>(_out.data_ptr());

        sample_conv_core(
            ptrA,
            ptrout,
            kernel_size,
            stride,
            C, H, W
        );
        
        if(!out.is_contiguous())
            out.copy_(_out);
    // Return the Torch tensor back to PyTorch
        return out;
    }
    TORCH_LIBRARY_IMPL(refiner_C, CUDA, m) {
        m.impl("sample_conv", &sample_conv);
    }
}

