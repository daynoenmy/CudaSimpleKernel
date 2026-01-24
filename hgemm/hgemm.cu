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
hgemm_baseline(half *x, half *y, half *out, int M, int N, int K)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    float v = 0;
#pragma unroll
    for (int i = 0; i < K; i++)
    {
        v += __half2float(x[m * K + i]) * __half2float(y[i * N + n]);
    }
    out[m * N + n] = __float2half(v);
}
template <const int BM, const int BN, const int BK>
__global__ void hgemm_v1(half *x, half *y, half *out, int M, int N, int K)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int tidx = threadIdx.x; // x -> N
    int tidy = threadIdx.y; // y -> M
    float v = 0.0f;
    __shared__ float smem_a[BM][BK];
    __shared__ float smem_b[BK][BN];
    const int title_k = K / BK;
    for (int i = 0; i < title_k; i++)
    {
        for (int k = 0; k < BK; k++)
        {
            smem_a[tidy][k] = __half2float(x[idy * K + i * BK + k]);
            smem_b[k][tidx] = __half2float(y[(i * BK + k) * N + idx]);
        }
        __syncthreads();

        for (int l = 0; l < BK; l++)
        {
            v += smem_a[tidy][l] * smem_b[l][tidx];
        }
        __syncthreads();
    }
    out[idy * N + idx] = __float2half(v);
}
template <const int BM, const int BN, const int BK>
__global__ void hgemm_float4(half *x, half *y, half *out, int M, int N, int K)
{
    const int stride = 2;
    int tidx = threadIdx.x*stride;
    int tidy = threadIdx.y*stride;
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
    int idy = (blockIdx.y * blockDim.y + threadIdx.y) * stride;
    float temp[stride][stride] = {0.0f};
    __shared__ float mem_a[BM][BK];
    __shared__ float mem_b[BK][BN];
    int title_size = K / BK;
    for (int i = 0; i < title_size; i++)
    {
        for (int k = 0; k < BK; k++)
        {
            mem_a[tidy][k] = __half2float(x[idy * K + i * BK + k]);
            mem_a[tidy + 1][k] = __half2float(x[(idy + 1) * K + i * BK + k]);
            // mem_a[tidy + 2][k] = __half2float(x[(idy + 2) * K + i * BK + k]);
            // mem_a[tidy + 3][k] = __half2float(x[(idy + 3) * K + i * BK + k]);
            mem_b[k][tidx] = __half2float(y[idx + (i * BK + k) * N]);
            mem_b[k][tidx + 1] = __half2float(y[idx + 1 + (i * BK + k) * N]);
            // mem_b[k][tidx+2] = __half2float(y[idx + 2 + (i * BK + k) * N]);
            // mem_b[k ][tidx+3] = __half2float(y[idx + 3 + (i * BK + k) * N]);
            __syncthreads();
        }
        for (int k = 0; k < BK; k++)
        {
            for (int r = 0; r < stride; r++)
            {
                for (int c = 0; c < stride; c++)
                {
                    temp[r][c] += mem_a[tidy + r][k] * mem_b[k][tidx + c];
                }
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < stride; i++)
    {
        for (int j = 0; j < stride; j++)
        {
            out[(idy + i) * N + idx + j] = __float2half(temp[i][j]);
        }
    }
}
torch::Tensor matrix_mul_baseline(torch::Tensor x, torch::Tensor y)
{
    const int block_size = 16;
    int M = x.size(0);
    int K = x.size(1);
    int N = y.size(1);
    torch::Tensor out = torch::zeros({M, N}, x.options());
    dim3 block(block_size, block_size);
    dim3 grid(N / block_size, M / block_size);
    at::Half *x_ptr = x.data_ptr<at::Half>();
    at::Half *y_ptr = y.data_ptr<at::Half>();
    at::Half *out_ptr = out.data_ptr<at::Half>();
    hgemm_baseline<<<grid, block>>>(reinterpret_cast<half *>(x_ptr),
                                    reinterpret_cast<half *>(y_ptr),
                                    reinterpret_cast<half *>(out_ptr), M, N, K);
    return out;
}

torch::Tensor matrix_mul_v1(torch::Tensor x, torch::Tensor y)
{
    const int block_size = 16;
    int M = x.size(0);
    int K = x.size(1);
    int N = y.size(1);
    torch::Tensor out = torch::zeros({M, N}, x.options());
    dim3 block(block_size, block_size);
    dim3 grid(N / block_size, M / block_size);
    at::Half *x_ptr = x.data_ptr<at::Half>();
    at::Half *y_ptr = y.data_ptr<at::Half>();
    at::Half *out_ptr = out.data_ptr<at::Half>();
    hgemm_v1<block_size, block_size, block_size><<<grid, block>>>(reinterpret_cast<half *>(x_ptr),
                                                                  reinterpret_cast<half *>(y_ptr),
                                                                  reinterpret_cast<half *>(out_ptr), M, N, K);
    return out;
}
torch::Tensor matrix_mul_float4(torch::Tensor x, torch::Tensor y)
{
    const int block_size = 16;
    int M = x.size(0);
    int K = x.size(1);
    int N = y.size(1);
    torch::Tensor out = torch::zeros({M, N}, x.options());
    dim3 block(block_size / 2, block_size / 2);
    dim3 grid(N / block_size, M / block_size);
    at::Half *x_ptr = x.data_ptr<at::Half>();
    at::Half *y_ptr = y.data_ptr<at::Half>();
    at::Half *out_ptr = out.data_ptr<at::Half>();
    hgemm_float4<block_size, block_size, block_size><<<grid, block>>>(reinterpret_cast<half *>(x_ptr),
                                                                      reinterpret_cast<half *>(y_ptr),
                                                                      reinterpret_cast<half *>(out_ptr), M, N, K);
    return out;
}