#ifndef RMSNORMAL_BASELINE_
#define RMSNORMAL_BASELINE_

#include<cstdio>

#define WARP_SIZE 32


template<const int WarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum(float val)
{
#pragma unroll
    for (int mask = WarpSize>>1; mask >=1 ; mask>>=1)
    {
        val += __shfl_xor_sync(0xffffffff,val,mask);
    }
    return val;
    
}
template<const int NUM_THREADS=256>
__device__ __forceinline__ float block_reduce_sum(float val)
{
    constexpr int NUM_WARPS = (NUM_THREADS+WARP_SIZE-1)/WARP_SIZE;
    int warpId = threadIdx.x/WARP_SIZE;
    int laneId = threadIdx.x%WARP_SIZE;
    static __shared__ float shared[NUM_WARPS];
    val = warp_reduce_sum<WARP_SIZE>(val);
    if(laneId==0)
    {
        shared[warpId] = val;
    }
    __syncthreads();
    val = (laneId < NUM_WARPS) ? shared[laneId]:0.0f;
    val = warp_reduce_sum<WARP_SIZE>(val);
    // 只有localid = 0时是整个块的sum
    return val;
}

// rmsnormal 
// 输入数据x : N*K =>batch_size*seq_len
// grid(N*K/K) block(K)
#define EPSILON 1e-5f
template<const int NUM_THREADS=256>
__global__ void rms_normal_baseline(float *x,float *y,float gamma,int N,int K)
{
    int tid = threadIdx.x;
    int bid =blockIdx.x;
    int idx = bid*blockDim.x + tid;
    __shared__ float s_variance;
    float value = idx<N*K?x[idx]:0.0f;
    float variance = value*value;
    variance = block_reduce_sum<NUM_THREADS>(variance);
    if(tid==0)
    {
        s_variance = rsqrtf(variance/(float)K+EPSILON);
    }
    __syncthreads();
    if(idx<N*K)
    {
        y[idx] = (value*s_variance)*gamma;
    }
}

template<const int NUM_THREADS>
void launch_kernel(float *x,float *y,float gamma,int N,int K)
{
    dim3 block(K);
    dim3 grid(N);
    rms_normal_baseline<NUM_THREADS><<<grid,block>>>(x,y,gamma,N,K);
}

template void launch_kernel<256>(float*, float*, float, int, int);
#endif