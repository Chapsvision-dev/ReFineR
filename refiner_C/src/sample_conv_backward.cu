#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#include "helpers.h"


#define BS 32

namespace refiner_C{
    __global__ void sample_conv_backward_kernel(
        float const *grad_A, // C x Hk x Wk x kx x ky
        float const *A,
        float *out,
        int const kernel_size, 
        int const stride,
        int C, int H, int W, int Hk, int Wk
    ){
        int x = threadIdx.x + BS * blockIdx.x;
        int y = threadIdx.y + BS * blockIdx.y;

        if (x >= W || y >= H){return;}

        float nx = lrintf(floorf(((float)x-kernel_size) / stride)) + 1;
        nx = max(nx, 0.);
        float ny = lrintf(floorf(((float)y-kernel_size) / stride)) + 1;
        ny = max(ny, 0.);

        if (x - nx * stride < 0 || y - ny * stride < 0){return;}
        if (nx * stride > W-kernel_size || ny * stride > H-kernel_size){return;}

        int mx = nx + lrintf(floorf((float)x / stride - nx));
        int my = ny + lrintf(floorf((float)y / stride - ny));

        float sum = 0.;
        // sum = nx +0.1*mx+0.01*ny+0.001*my;
        int d0 = kernel_size; 
        int d1 = kernel_size * d0;
        int d2 = Wk * d1;
        int d3 = Hk * d2;
        for (int i = nx; i <= mx; i++){
            if (i * stride > W - kernel_size){continue;} // Maybe recompute mx and my to avoid these if statements. Work for future me.
            for (int j = ny; j <= my; j++){
                if (j * stride > H - kernel_size){continue;}
                sum += grad_A[(x - i * stride) + (y - j * stride) * d0 + i * d1 + j * d2 + blockIdx.z * d3];
            }
        }

        out[x + y * W + blockIdx.z * H * W] = sum;
    }
    void sample_conv_backward_core(
        float const *grad_A,
        float const *A,
        float *out,
        int const kernel_size,
        int const stride,
        int C, int H, int W, int Hk, int Wk
    ){  
        dim3 dimGrid;
        dimGrid.x = (W + BS - 1) / BS;
        dimGrid.y = (H + BS - 1) / BS;
        dimGrid.z = C;

        dim3 dimBlock;
        dimBlock.x = BS;
        dimBlock.y = BS;
        dimBlock.z = 1;

        sample_conv_backward_kernel<<<dimGrid, dimBlock>>>(grad_A, A, out, kernel_size, stride, C, H, W, Hk, Wk);
    }
    

    torch::Tensor sample_conv_backward(
        const torch::Tensor grad_A, // C x (H - k)/s + 1 x (W - k)/s + 1 x k x k
        const torch::Tensor A, // C x H x W
        int64_t const kernel_size,
        int64_t const stride
    ) {  
        // tensor sample with stride 1
        const int W = A.sizes()[2];
        const int H = A.sizes()[1];
        const int C = A.sizes()[0];
        const int Hk = grad_A.sizes()[1];
        const int Wk = grad_A.sizes()[2];


        auto out_options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
        torch::Tensor out;
        out = torch::full({C, H, W}, 0., out_options);

        torch::Tensor _grad_A = grad_A.contiguous();
        torch::Tensor _A = A.contiguous();
        torch::Tensor _out = out.contiguous();
    
        float const *ptrgrad_A = reinterpret_cast<float*>(_grad_A.data_ptr());
        float const *ptrA = reinterpret_cast<float*>(_A.data_ptr());
        float *ptrout = reinterpret_cast<float*>(_out.data_ptr());

        sample_conv_backward_core(
            ptrgrad_A,
            ptrA,
            ptrout,
            kernel_size,
            stride,
            C, H, W, Hk, Wk
        );
        
        if(!out.is_contiguous())
            out.copy_(_out);

    // Return the Torch tensor back to PyTorch
        return out;
    }
    TORCH_LIBRARY_IMPL(refiner_C, CUDA, m) {
        m.impl("sample_conv_backward", &sample_conv_backward);
    }
}

