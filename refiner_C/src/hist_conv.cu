#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#include "helpers.h"

#define BS 16
#define NUM_BINS 16

__constant__ float const_bins[NUM_BINS];

namespace refiner_C{
    __global__ void hist_conv_kernel(
        float const *A,
        float const *weights,
        float *out,
        int const kernel_size,
        float const sigmam1, 
        int const C, int const H, int const W, int const B,
        int const kbs, int const Wk, int const Hk,
        int const mW, int const mH, int const M, int const bs
    ){
        extern __shared__ float sbins[]; // M x M x BS x BS x B
        int d0 = B;
        int d1 = bs * d0;
        int d2 = bs * d1;
        int d3 = M * d2;

        int Mx = (threadIdx.z % M);
        int My = (threadIdx.z / M);

        int outx = (Mx + blockIdx.x * M);
        int outy = (My + blockIdx.y * M);

        if(outx >= Wk || outy >= Hk){return;}

        int ob = threadIdx.x * d0 + threadIdx.y * d1 + Mx * d2 + My * d3;
        for (int b = 0; b < B; b++){
            sbins[b + ob] = 0.;
        }

        float data;
        float weight;

        int ox = mW + blockIdx.x * kernel_size * M + Mx * kernel_size + threadIdx.x;
        int oy = mH + blockIdx.y * kernel_size * M + My * kernel_size + threadIdx.y;
        int ac = blockIdx.z;
        int ax;
        int ay;
        for (int i = 0; i < kbs; i++){
            for (int j = 0; j < kbs; j++){
                ax = ox + i * bs;
                ay = oy + j * bs;
                if (ax < W && ay < H){
                    data = A[ax + ay * W + ac * W * H];
                    weight = weights[ax + ay * W + ac * W * H];

                    for (int b = 0; b < B; b++){
                        sbins[b + ob] += weight * f_float(data, const_bins[b], sigmam1);
                    }
                }
                else{
                    return;
                }
            }
        }

        // H reduction
        for (int redy = bs / 2; redy > 0; redy >>= 1) {
            __syncthreads();
            if (threadIdx.y < redy){
                for (int b = 0; b < B; b++){
                    sbins[b + ob] += sbins[b + ob + redy * d1];
                }
            }
            else{
                return; // Useless threads terminate
            }
        }

        // W reduction
        for (int redx = bs / 2; redx > 0; redx >>= 1) {
            __syncthreads();
            if (threadIdx.x < redx){
                for (int b = 0; b < B; b++){
                    sbins[b + ob] += sbins[b + ob + redx * d0];
                }
            }
            else{
                return; // Useless threads terminate
            }
        }
        __syncthreads();
        if (threadIdx.x == 0 && threadIdx.y == 0){
            for (int b = 0; b < B; b++){
                out[b + outx * B + outy * B * Wk + blockIdx.z * B * Wk * Hk]
                = sbins[b + Mx * d2 + My * d3];
            }
        }
    }
    void hist_conv_core(
        float const *A,
        float const *weights,
        float *out,
        int const kernel_size,
        float const sigma,
        int const C, int const H, int const W, int const B
    ){  
        int Wk = W / kernel_size;
        int Hk = H / kernel_size;
        int kbs;
        int M;
        int bs;

        if (kernel_size < BS){
            kbs = 1;
            M = 2 * BS / kernel_size;
            bs = kernel_size;
        }
        else{
            M = 2;
            kbs = kernel_size / BS;
            bs = BS;
        }

        int mW = (W - Wk * kernel_size) / 2;
        int mH = (H - Hk * kernel_size) / 2;
        dim3 dimGrid;
        dimGrid.x = (Wk + M - 1) / M;
        dimGrid.y = (Hk + M - 1) / M;
        dimGrid.z = C;

        dim3 dimBlock;
        dimBlock.x = bs;
        dimBlock.y = bs;
        dimBlock.z = M * M;

        hist_conv_kernel<<<dimGrid, dimBlock, M * M * bs * bs * B * sizeof(float)  >>>(A, weights, out, kernel_size, 1/sigma, C, H, W, B, kbs, Wk, Hk, mW, mH, M, bs);

    }
    

    torch::Tensor hist_conv(
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

        if(!(A.device().is_cuda() && bins.device().is_cuda()))
            throw std::invalid_argument("wrapper only supports GPU device. Use .to(device=torch.device('cuda'))");
    
        if (W < kernel_size || H < kernel_size)
            throw std::invalid_argument("Width or height smaller than kernel size");

        if (B > NUM_BINS){
            throw std::invalid_argument("Maximum number of bins exceeded");
        }
        
        if (A.dtype() != torch::kFloat32 || bins.dtype() != torch::kFloat32)
            throw std::invalid_argument("Tensors A and bins should both have float32 type");

        auto out_options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
        torch::Tensor out;
        out = torch::full({C, H / kernel_size, W / kernel_size, B}, 0., out_options);

        torch::Tensor _A = A.contiguous();
        torch::Tensor _weights = weights.contiguous();
        torch::Tensor _bins = bins.contiguous();
        torch::Tensor _out = out.contiguous();
    
        float const *ptrA = reinterpret_cast<float*>(_A.data_ptr());
        float const *ptrweights = reinterpret_cast<float*>(_weights.data_ptr());
        float *ptrout = reinterpret_cast<float*>(_out.data_ptr());
        float *ptrbins = reinterpret_cast<float*>(_bins.data_ptr());
        
        // cudaMemcpy(bins, ptrbins, B * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpyToSymbol(const_bins, ptrbins, B * sizeof(float));
        hist_conv_core(
            ptrA,
            ptrweights,
            ptrout,
            kernel_size,
            sigma,
            C, H, W, B
        );
        
        if(!out.is_contiguous())
            out.copy_(_out);

    // Return the Torch tensor back to PyTorch
        return out;
    }

    TORCH_LIBRARY_IMPL(refiner_C, CUDA, m) {
        m.impl("hist_conv", &hist_conv);
    }
}

