#include<cstdio>
#include<cuda.h>
#include<cuda_runtime.h>
#define Warp_Size 32
__device__ __forceinline__ float warp_reduce_sum(float value)
{
    for (int mask = Warp_Size>>1; mask>=1; mask>>=1)
    {
        value += __shfl_xor_sync(0xffffffff,value,mask);
    }
    return value;
    
}
template<const int block_size>
__device__ __forceinline__ float block_reduce_sum(float value)
{
    constexpr int warp_num =block_size /Warp_Size;
    int warpId = threadIdx.x/Warp_Size;
    int laneId = threadIdx.x%Warp_Size;
    static __shared__ float s_data[warp_num];
    value = warp_reduce_sum(value);
    if(laneId==0){
        s_data[warpId] = value;
    }
    __syncthreads();
    value = laneId<warp_num?s_data[laneId]:0.0f;
    value = warp_reduce_sum(value);
    return value;
}
template<const int block_size>
__global__ void rms_normal_v1(float *x,float *y,float *gmma,const int N,const int K)
{
    
}
