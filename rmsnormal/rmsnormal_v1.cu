#ifndef RMS_NORMAL_V1_
#define RMS_NORMAL_V1_
#include<cstdio>
#include<cuda.h>
#include<cuda_runtime.h>
#define Warp_Size 32
__device__ __forceinline__ float warp_reduce_sum(float value)
{
#pragma unroll
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
__global__ void rms_normal_v1(float *x,float *y,float gamma,const int N,const int K)
{
    const float epsilon = 1e-5f;
    int tid =  threadIdx.x;
    int bid = blockIdx.x;
    int gid = bid*blockDim.x+tid;
    __shared__ float s_variance;
    float4 reg_x = (reinterpret_cast<float4 *>(&x[gid*4]))[0];
    float variance = (gid*4 <N*K) ? (reg_x.x*reg_x.x + reg_x.y*reg_x.y+reg_x.z*reg_x.z+reg_x.w*reg_x.w):0.0f;
    variance = block_reduce_sum<block_size>(variance);
    if(tid==0)
    {
        s_variance = rsqrtf(variance/(float)K+epsilon);
    }
    __syncthreads();
    float4 reg_y;
    reg_y.x = reg_x.x*s_variance*gamma;
    reg_y.y = reg_x.y*s_variance*gamma;
    reg_y.z = reg_x.z*s_variance*gamma;
    reg_y.w = reg_x.w*s_variance*gamma;
    if(gid*4<N*K){
        (reinterpret_cast<float4 *>(&y[gid*4]))[0] = reg_y;
    }
    
}
template<const int block_size>
void launch_kernel(float *x,float *y,float gamma,int N,int K)
{
    dim3 block(block_size);
    dim3 grid(N);
    rms_normal_v1<block_size><<<grid,block>>>(x,y,gamma,N,K);
}
template void launch_kernel<64>(float *x,float *y,float gamma,int N,int K);
#endif