#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#include "helpers.h"


#define BS 4
#define NUM_BINS 16

namespace refiner_C{
   
    __global__ void hist_conv_backward_kernel(
        float const *grad_output,
        float const *A,
        float const *weights,
        float *grad_A,
        float *grad_weights,
        float const *bins,
        int const kernel_size,
        float const sigmam1, 
        int const C, int const H, int const W, int const B,
        int const kbs, int const Wk, int const Hk,
        int const mW, int const mH
    ){
        int ax = mW + threadIdx.x + blockIdx.x * BS;
        int ay = mH + threadIdx.y + blockIdx.y * BS;
        int ac = blockIdx.z;

        __shared__ float sdata[BS][BS];
        __shared__ float sweights[BS][BS];
        __shared__ float sbins[NUM_BINS];
        __shared__ float sgrad[NUM_BINS];
        __shared__ float sgrad_A[NUM_BINS][BS][BS];
        __shared__ float sgrad_weights[NUM_BINS][BS][BS];

        if (threadIdx.z == 0){
            sdata[threadIdx.y][threadIdx.x] = A[ax + W * ay + H * W * ac];
            sweights[threadIdx.y][threadIdx.x] = weights[ax + W * ay + H * W * ac];
        }
        // Loading bins and gradient outputs
        if (threadIdx.x == 0 && threadIdx.y == 0){
            sbins[threadIdx.z] = bins[threadIdx.z]; // Not coalesced probably better with constant memory :(
            sgrad[threadIdx.z] = grad_output[threadIdx.z + B * (blockIdx.x / kbs) + B * Wk * (blockIdx.y / kbs) + B * Wk * Hk * blockIdx.z];
        }

        __syncthreads();
        sgrad_A[threadIdx.z][threadIdx.y][threadIdx.x] = fp(sdata[threadIdx.y][threadIdx.x], sbins[threadIdx.z], sigmam1) * sgrad[threadIdx.z] * sweights[threadIdx.y][threadIdx.x];
        sgrad_weights[threadIdx.z][threadIdx.y][threadIdx.x] = f_float(sdata[threadIdx.y][threadIdx.x], sbins[threadIdx.z], sigmam1) * sgrad[threadIdx.z];

        // B reduction
        #pragma unroll
        for (int redb = B / 2; redb > 0; redb >>= 1) {
            __syncthreads();
            if (threadIdx.z < redb){
            sgrad_A[threadIdx.z][threadIdx.y][threadIdx.x] += sgrad_A[threadIdx.z + redb][threadIdx.y][threadIdx.x];
            sgrad_weights[threadIdx.z][threadIdx.y][threadIdx.x] += sgrad_weights[threadIdx.z + redb][threadIdx.y][threadIdx.x];
            
            }
            else{
                return; // Useless threads terminate
            }
        }

        grad_A[ax + ay * W + ac * W * H] = sgrad_A[threadIdx.z][threadIdx.y][threadIdx.x];
        grad_weights[ax + ay * W + ac * W * H] = sgrad_weights[threadIdx.z][threadIdx.y][threadIdx.x];
    }

    void hist_conv_backward_core(
        float const *grad_output,
        float const *A,
        float const *weights,
        float *grad_A,
        float *grad_weights,
        float const* bins,
        int const kernel_size,
        float const sigma,
        int const C, int const H, int const W, int const B
    ){

        int kbs = kernel_size / BS;
        int Wk = W / kernel_size;
        int Hk = H / kernel_size;

        int mW = (W - Wk * kernel_size) / 2;
        int mH = (H - Hk * kernel_size) / 2;

        dim3 dimGrid;
        dimGrid.x = Wk * kbs;
        dimGrid.y = Hk * kbs;
        dimGrid.z = C;

        dim3 dimBlock;
        dimBlock.x = BS;
        dimBlock.y = BS;
        dimBlock.z = B;
        
        hist_conv_backward_kernel<<<dimGrid, dimBlock>>>(
            grad_output, A, weights, grad_A, grad_weights, bins, kernel_size, 1 / sigma,
            C, H, W, B, kbs, Wk, Hk, mW, mH);
    }
    
    std::tuple<torch::Tensor, torch::Tensor> hist_conv_backward(
        const torch::Tensor grad_output, // C x H/K x W/K x B
        const torch::Tensor A, // C x H x W
        const torch::Tensor weights, // C x H x W
        const torch::Tensor bins, // B
        int64_t kernel_size,
        double sigma
    ) {  
        const int W = A.sizes()[2];
        const int H = A.sizes()[1];
        const int C = A.sizes()[0];
        const int B = bins.sizes()[0];

        if(!(A.device().is_cuda() && weights.device().is_cuda() && grad_output.device().is_cuda() && bins.device().is_cuda()))
            throw std::invalid_argument("wrapper only supports GPU device. Use .to(device=torch.device('cuda'))");
    
        if (W < kernel_size || H < kernel_size)
            throw std::invalid_argument("Width or height smaller than kernel size");

        if (B > NUM_BINS || kernel_size < BS){
            throw std::invalid_argument("Invalid bin length or kernel_size");
        }
        
        if (A.dtype() != torch::kFloat32 || weights.dtype() != torch::kFloat32 || grad_output.dtype() != torch::kFloat32 || bins.dtype() != torch::kFloat32)
            throw std::invalid_argument("Tensors A, weights, grads,  bins should both have float32 type");

        auto out_options = torch::TensorOptions().device(torch::kCUDA).dtype(A.dtype());
        torch::Tensor grad_A;
        torch::Tensor grad_weights;
        grad_A = torch::full({C, H, W}, 0., out_options);
        grad_weights = torch::full({C, H, W}, 0., out_options);

        torch::Tensor _grad_output = grad_output.contiguous();
        torch::Tensor _A = A.contiguous();
        torch::Tensor _weights = weights.contiguous();
        torch::Tensor _bins = bins.contiguous();
        torch::Tensor _grad_A = grad_A.contiguous();
        torch::Tensor _grad_weights = grad_weights.contiguous();
        
        float const *ptrgrad_output = reinterpret_cast<float*>(_grad_output.data_ptr());
        float const *ptrA = reinterpret_cast<float*>(_A.data_ptr());
        float const *ptrweights = reinterpret_cast<float*>(_weights.data_ptr());
        float *ptrgrad_A = reinterpret_cast<float*>(_grad_A.data_ptr());
        float *ptrgrad_weights = reinterpret_cast<float*>(_grad_weights.data_ptr());
        float const *ptrbins = reinterpret_cast<float*>(_bins.data_ptr());

        hist_conv_backward_core(
            ptrgrad_output,
            ptrA,
            ptrweights,
            ptrgrad_A,
            ptrgrad_weights,
            ptrbins,
            kernel_size,
            sigma,
            C, H, W, B
        );
        
        grad_A.copy_(_grad_A);
        grad_weights.copy_(_grad_weights);

        return std::make_tuple(grad_A, grad_weights);
    }

    TORCH_LIBRARY_IMPL(refiner_C, CUDA, m) {
        m.impl("hist_conv_backward", &hist_conv_backward);
    }
}

