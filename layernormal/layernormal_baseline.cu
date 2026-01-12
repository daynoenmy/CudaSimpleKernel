#include<cstdio>
#include<cuda.h>
#include<cuda_runtime.h>
#define Warp_Size 32
#define Warp_All_Thread 0xffffffff
__device__ __forceinline__ float warp_reduce_sum(float value)
{
    for (int mask = Warp_Size>>1; mask >0; mask>>=1)
    {
        value += __shfl_xor_sync(Warp_All_Thread,value,mask);
    }
    return value;
}
template<const int block_size>
__device__ float block_reduce_sum(float value)
{
    constexpr int warp_num = block_size/Warp_Size;
    int warpId = threadIdx.x/Warp_Size;
    int laneId = threadIdx.x%Warp_Size;
    static __shared__ float s_data[warp_num];
    value = warp_reduce_sum(value); 
    if(laneId==0)
        s_data[warpId] = value;
    __syncthreads();
    value = laneId<warp_num?s_data[laneId]:0.0f;
    value = warp_reduce_sum(value);
    return value;
}
template<const int block_size>
__global__ void layer_normal_baseline(float *x,float *y,float gamma,const int N,const int K)
{
    const float epsilon = 1e-5f;
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int idx = bid*block_size + tid;
    __shared__ float s_variance;
    __shared__ float s_mean;
    float variance=idx<N*K?x[idx]:0.0f;
    float sum = block_reduce_sum<block_size>(variance);
    if(tid==0)
        s_mean = sum/float(K);
    
    __syncthreads();
   
    float v = (variance-s_mean)*(variance-s_mean);
    v  = block_reduce_sum<block_size>(v);
    if(tid==0)
        s_variance = rsqrtf(v/float(K)+epsilon);
    
    __syncthreads();
    
    if(idx < N*K)
        y[idx] = (variance-s_mean)*s_variance*gamma;
}
template<const int NUM_THREADS>
void lanuch_kernel(float *x,float *y,float gamma, const int N,const int K)
{
    dim3 block(K);
    dim3 grid(N);
    layer_normal_baseline<NUM_THREADS><<<grid,block>>>(x,y,gamma,N,K);
}
template void  lanuch_kernel<256>(float *x,float *y,float gamma, const int N,const int K);