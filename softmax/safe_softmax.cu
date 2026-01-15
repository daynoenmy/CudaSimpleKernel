#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <torch/extension.h>
#include <torch/types.h>
#define Warp_Size 32
#define WARP_ALL_THREAD 0xffffffff
template <const int WarpSize>
__device__ __forceinline__ float warp_reduce_sum(float value)
{
    for (int mask = WarpSize >> 1; mask > 0; mask >>= 1)
    {
        value += __shfl_xor_sync(WARP_ALL_THREAD, value, mask);
    }
    return value;
}
template <const int WarpSize>
__device__ __forceinline__ float warp_reduce_max(float value)
{
    for (int mask = WarpSize >> 1; mask > 0; mask >>= 1)
    {
        value = fmaxf(value, __shfl_xor_sync(WARP_ALL_THREAD, value, mask));
    }
    return value;
}
template <const int block_size>
__device__ float block_reduce_sum(float value)
{
    constexpr int warp_num = block_size / Warp_Size;
    int warpId = threadIdx.x / Warp_Size;
    static __shared__ float s_data[warp_num];
    int laneId = threadIdx.x % Warp_Size;
    value = warp_reduce_sum<Warp_Size>(value);
    if (laneId == 0)
    {
        s_data[warpId] = value;
    }
    __syncthreads();
    value = laneId < warp_num ? s_data[laneId] : 0.0f;
    value = warp_reduce_sum<Warp_Size>(value);
    value = __shfl_sync(WARP_ALL_THREAD, value, 0);
    return value;
}
template <const int block_size>
__device__ float block_reduce_max(float value)
{
    constexpr int warp_num = block_size / Warp_Size;
    int warpId = threadIdx.x / Warp_Size;
    static __shared__ float s_data[warp_num];
    int laneId = threadIdx.x % Warp_Size;
    value = warp_reduce_max<Warp_Size>(value);
    if (laneId == 0)
    {
        s_data[warpId] = value;
    }
    __syncthreads();
    value = laneId < warp_num ? s_data[laneId] : -INFINITY;
    value = warp_reduce_max<Warp_Size>(value);
    value = __shfl_sync(WARP_ALL_THREAD, value, 0);
    return value;
}
template <const int block_size>
__global__ void safe_softmax_f32x4(float *x, float *y)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = (bid * block_size + tid) * 4;
    float4 reg_x = (reinterpret_cast<float4 *>(&x[idx]))[0];
    float max_val = fmaxf(reg_x.x, fmaxf(reg_x.y, fmaxf(reg_x.z, reg_x.w)));
    max_val = block_reduce_max<block_size>(max_val);
    float4 reg_exp;
    reg_exp.x = expf(reg_x.x - max_val);
    reg_exp.y = expf(reg_x.y - max_val);
    reg_exp.z = expf(reg_x.z - max_val);
    reg_exp.w = expf(reg_x.w - max_val);
    float exp_sum = reg_exp.x + reg_exp.y + reg_exp.z + reg_exp.w;
    exp_sum = block_reduce_sum<block_size>(exp_sum);
    float4 reg_y;
    reg_y.x = reg_exp.x / exp_sum;
    reg_y.y = reg_exp.y / exp_sum;
    reg_y.w = reg_exp.w / exp_sum;
    reg_y.z = reg_exp.z / exp_sum;
    (reinterpret_cast<float4 *>(&y[idx]))[0] = reg_y;
}
torch::Tensor safe_softmax_forward(torch::Tensor x)
{
    torch::Tensor y = torch::zeros_like(x);
    const int N = 256;
    const int K =512;
    dim3 block(K/4);
    dim3 grid(N);
    safe_softmax_f32x4<K/4><<<grid,block>>>(x.data_ptr<float>(),y.data_ptr<float>());
    return y;
}