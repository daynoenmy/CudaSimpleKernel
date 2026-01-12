#ifndef LAYER_NORMAL_V1_
#define LAYER_NORMAL_V1_
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#define debug 0
#define Warp_Size 32
#define Warp_All_Threads 0xffffffff
__device__ __forceinline__ float warp_reduce_sum(float value)
{
    for (int mask = Warp_Size >> 1; mask > 0; mask >>= 1)
    {
        value += __shfl_xor_sync(Warp_All_Threads, value, mask);
    }
    return value;
}
template <const int block_size>
__device__ float block_reduce_sum(float value)
{
    constexpr int warp_num = block_size / Warp_Size;
    int warpId = threadIdx.x / Warp_Size;
    int laneId = threadIdx.x % Warp_Size;
    static __shared__ float s_data[warp_num];
    value = warp_reduce_sum(value);
    if (laneId == 0)
    {
        s_data[warpId] = value;
    }
    __syncthreads();
    value = laneId < warp_num ? s_data[laneId] : 0.0f;
    value = warp_reduce_sum(value);
    return value;
}
template <const int block_size>
__global__ void layer_normal_v1(float *x, float *y, float gamma, const int N, const int K)
{
    const float epslion = 1e-5f;
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = (tid + bid * block_size) * 4;
    __shared__ float s_mean;
    __shared__ float s_variance;
    float4 reg_x = (reinterpret_cast<float4 *>(&x[idx]))[0];
    float value = reg_x.x + reg_x.y + reg_x.z + reg_x.w;
    value = block_reduce_sum<block_size>(value);
    if (tid == 0)
    {
        s_mean = value / (float)K;
    }
    __syncthreads();
#if debug
    if (tid == 0 && bid == 0)
    {
        printf("s_mean=%f\n", s_mean);
    }
#endif
    float variance = (reg_x.x - s_mean) * (reg_x.x - s_mean) + (reg_x.y - s_mean) * (reg_x.y - s_mean) +
                     (reg_x.z - s_mean) * (reg_x.z - s_mean) +
                     (reg_x.w - s_mean) * (reg_x.w - s_mean);
    variance = block_reduce_sum<block_size>(variance);
    if (tid == 0)
    {
        s_variance = rsqrtf(variance / (float)K + epslion);
    }
    __syncthreads();
#if debug
    if (tid == 0 && bid == 0)
    {
        printf("s_variance=%f\n", s_variance);
    }
#endif
    float4 reg_y;
#if debug
    if (tid == 0 && bid == 0)
        printf("reg_y.x = %f\n", reg_x.x);
#endif
    reg_y.x = gamma * (reg_x.x - s_mean) * s_variance;
    reg_y.y = gamma * (reg_x.y - s_mean) * s_variance;
    reg_y.z = gamma * (reg_x.z - s_mean) * s_variance;
    reg_y.w = gamma * (reg_x.w - s_mean) * s_variance;
    if (idx < N * K)
        (reinterpret_cast<float4 *>(&y[idx]))[0] = reg_y;
}
template <const int NUM_THREADS>
void lanuch_kernel(float *x, float *y, float gamma, const int N, const int K)
{
    dim3 block(NUM_THREADS / 4);
    dim3 grid(N);
    layer_normal_v1<NUM_THREADS / 4><<<grid, block>>>(x, y, gamma, N, K);
}
template void lanuch_kernel<256>(float *x, float *y, float gamma, const int N, const int K);
#endif