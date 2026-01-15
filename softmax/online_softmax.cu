#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/types.h>
#define WARP_SIZE 32
#define WARP_ALL_THREADS 0xffffffff
void online_softmax_cpu(float *x, float *y, int B, int S)
{
    for (int j = 0; j < B; j++)
    {
        float max_value = 0.0f;
        float sum = 0.0f;
        float pre_value = 0.0f;
        for (int i = 0; i < S; i++)
        {
            max_value = fmaxf(x[i + j * S], -INFINITY);
            sum = sum * expf(pre_value - max_value) + expf(x[i + j * S] - max_value);
            pre_value = max_value;
        }
        for (int i = 0; i < S; i++)
        {
            y[i + j * S] = expf(x[i + j * S] - max_value) / sum;
        }
    }
}
struct __align__(8) MD
{
    float m;
    float d;
};
template <const int WarpSize = WARP_SIZE>
__device__ __forceinline__ MD warp_reduce_md_op(MD value)
{
#pragma unroll
    for (int stride = WARP_SIZE >> 1; stride > 0; stride >>= 1)
    {
        MD other;
        other.m = __shfl_xor_sync(WARP_ALL_THREADS, value.m, stride);
        other.d = __shfl_xor_sync(WARP_ALL_THREADS, value.d, stride);
        bool value_bigger = (value.m > other.m);
        MD bigger_m = value_bigger ? value : other;
        MD smaller_m = value_bigger ? other : value;
        value.d = smaller_m.d * expf(smaller_m.m - bigger_m.m) + bigger_m.d;
        value.m = bigger_m.m;
    }
    return value;
}
template <const int block_size = 256>
__global__ void online_softmax_f32_gpu(float *x, float *y)
{
    int local_tid = threadIdx.x;
    int global_tid = blockIdx.x * block_size + local_tid;
    constexpr int WARP_NUM = block_size / WARP_SIZE;
    int warp_id = local_tid / WARP_SIZE;
    int lane_id = local_tid % WARP_SIZE;
    MD val;
    val.m = x[global_tid];
    val.d = 1.0f;
    __shared__ MD shared[WARP_NUM];
    MD res = warp_reduce_md_op<WARP_SIZE>(val);
    if (lane_id == 0)
        shared[warp_id] = res;
    __syncthreads();
    if (local_tid < WARP_SIZE)
    {
        MD block_res = shared[local_tid];
        block_res = warp_reduce_md_op<WARP_NUM>(block_res);
        if (local_tid == 0)
        {
            shared[0] = block_res;
        }
    }
    __syncthreads();
    MD final_res = shared[0];
    float d_total_inverse = __fdividef(1.0f, final_res.d);
    y[global_tid] = __expf(x[global_tid] - final_res.m) * d_total_inverse;
}
template <const int block_size>
__global__ void
online_softmax_f32x4_gpu(float *x, float *y)
{
    int local_tid = threadIdx.x;
    int global_tid = (blockIdx.x * block_size + local_tid) * 4;
    constexpr int WARP_NUM = block_size / WARP_SIZE;
    int lane_id = local_tid % WARP_SIZE;
    int warp_id = local_tid / WARP_SIZE;
    float4 val = (reinterpret_cast<float4 *>(&x[global_tid]))[0];
    float local_m = fmaxf(fmaxf(val.x, val.y), fmaxf(val.z, val.w));
    float local_d = __expf(val.x - local_m) + __expf(val.y - local_m) + __expf(val.w - local_m) + __expf(val.z - local_m);
    MD local_md = {local_m, local_d};
    __shared__ MD shared[WARP_NUM];
    MD res = warp_reduce_md_op<WARP_SIZE>(local_md);
    if (lane_id == 0)
        shared[warp_id] = res;
    __syncthreads();
    if (local_tid < WARP_SIZE)
    {
        MD block_res = shared[local_tid];
        block_res = warp_reduce_md_op<WARP_NUM>(block_res);
        if (local_tid == 0)
        {
            shared[0] = block_res;
        }
    }
    __syncthreads();
    MD final_res = shared[0];
    float d_total_inverse = __fdividef(1.0f, final_res.d);
    float4 reg_y;
    reg_y.x = __expf(val.x - final_res.m) * d_total_inverse;
    reg_y.y = __expf(val.y - final_res.m) * d_total_inverse;
    reg_y.z = __expf(val.z - final_res.m) * d_total_inverse;
    reg_y.w = __expf(val.w - final_res.m) * d_total_inverse;
    (reinterpret_cast<float4*>(&y[global_tid]))[0] = reg_y;
}
torch::Tensor online_softmax_cpu_forward(torch::Tensor x)
{
    torch::Tensor out = torch::zeros_like(x);
    online_softmax_cpu(x.data_ptr<float>(), out.data_ptr<float>(), x.size(0), x.size(1));
    return out;
}
torch::Tensor online_softmax_f32_forward(torch::Tensor x)
{
    torch::Tensor out = torch::zeros_like(x);
    const int N = 512;
    const int H = 256;
    dim3 block(H);
    dim3 grid(N);
    online_softmax_f32_gpu<H><<<grid, block>>>(x.data_ptr<float>(), out.data_ptr<float>());
    return out;
}
torch::Tensor online_softmax_f32x4_forward(torch::Tensor x)
{
    torch::Tensor out = torch::zeros_like(x);
    const int N = 512;
    const int H = 256;
  
    dim3 grid(N);
      dim3 block(H/4);
    online_softmax_f32x4_gpu<H/4><<<grid, block>>>(x.data_ptr<float>(), out.data_ptr<float>());
    return out;
}
