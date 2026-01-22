#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <torch/types.h>
// M,K,N
// M N
// block(16,16)

__global__ void 
hgemm_baseline(half *x, half *y,half *out,int M,int N,int K)
{
    int n = blockIdx.x *blockDim.x + threadIdx.x;
    int m = blockIdx.y*blockDim.y + threadIdx.y;
    float v =0;
#pragma unroll
    for (int i = 0; i < K; i++)
    {
        v+= __half2float(x[m*K+i])*__half2float(y[i*N+n]); 
    }
    out[m*N+n] = __float2half(v);
}
torch::Tensor matrix_mul_baseline(torch::Tensor x,torch::Tensor y)
{
    const int block_size = 16;
    int M = x.size(0);
    int K = x.size(1);
    int N = y.size(1);
    torch::Tensor out = torch::zeros({M,N},x.options());
    dim3 block(block_size,block_size);
    dim3 grid(N/block_size,M/block_size);
     at::Half* x_ptr = x.data_ptr<at::Half>();
     at::Half* y_ptr = y.data_ptr<at::Half>();
    at::Half* out_ptr = out.data_ptr<at::Half>();
    hgemm_baseline<<<grid,block>>>( reinterpret_cast<half*>(x_ptr),
        reinterpret_cast< half*>(y_ptr),
        reinterpret_cast<half*>(out_ptr),M,N,K);
    return out;
    
}