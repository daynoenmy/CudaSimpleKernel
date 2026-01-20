#include <cstdio>
#include <thrust/device_vector.h>
#include <vector>
#include <cuda.h>
#include <curand.h>
#include <cuda_runtime.h>
#define WARP_SIZE 32
#define WARP_ALL_THREADS 0xffffffff
struct __align__(16) MD
{
    float m;
    float d;
    float o;
};
template <const int Warp_Size>
__device__ MD warp_reduce_op(MD value)
{
    MD other;
    for (int mask = Warp_Size >> 1; mask > 0; mask >>= 1)
    {
        other.m = __shfl_xor_sync(WARP_ALL_THREADS, value.m, mask);
        other.d = __shfl_xor_sync(WARP_ALL_THREADS, value.d, mask);
        other.o = __shfl_xor_sync(WARP_ALL_THREADS, value.o, mask);
        bool value_bigger = value.m > other.m;
        MD bigger = value_bigger ? value : other;
        MD smaller = value_bigger ? other : value;
        value.m = bigger.m;
        value.d = smaller.d * expf(smaller.m - bigger.m) + bigger.d;
        value.o = 1 / value.d * (smaller.d * expf(smaller.m - bigger.m) * smaller.o + bigger.d * bigger.o);
    }
    return value;
}
template <const int Block_Size>
__global__ void softmax_fused_dot(float *mat, float *vec, float *out, int N, int S)
{
    constexpr int warp_num = Block_Size / WARP_SIZE;
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int gid = blockIdx.x * Block_Size + tid;
    float data = mat[gid];
    int warpId = tid / WARP_SIZE;
    int laneId = tid % WARP_SIZE;
    __shared__ MD s_data[warp_num];
    MD value;
    value.m = data;
    value.d = 1.0f;
    value.o = vec[tid];
    value = warp_reduce_op<WARP_SIZE>(value);
    if (laneId == 0)
        s_data[warpId] = value;
    __syncthreads();
    if (tid < WARP_SIZE)
    {
        MD bloc_res = s_data[tid];
        bloc_res = warp_reduce_op<warp_num>(bloc_res);
        if (tid == 0)
        {
            s_data[0] = bloc_res;
        }
    }
    __syncthreads();
    MD  v = s_data[0];
    if (tid == 0)
    {
        out[bid] = v.o;
    }
}
void naive(float *input, float *vec, float *out, int N, int S)
{
    for (int j = 0; j < N; j++)
    {
        float max = input[j * S];
        float pre_max = -1000000;
        float sum = 0.0f;
        for (int i = 0; i < S; i++)
        {
            max = fmaxf(input[j * S + i], pre_max);
            sum = expf(input[j * S + i] - max) + sum * expf(pre_max - max);
            pre_max = max;
        }
        for (int i = 0; i < S; i++)
        {
            input[j * S + i] = expf(input[j * S + i] - max) / sum;
        }
        for (int i = 0; i < S; i++)
        {
          
            out[j] += input[j * S + i] * vec[i];
        }
    }
}
bool compare_res(float *cpu,float *gpu,int len,float diff)
{
    for (int i = 0; i <len; i++)
    {
        float d = abs(cpu[i]-gpu[i]);
        if(d>diff){
            printf("diff=%f,cpu=%f,gpu=%f\n",d,cpu[i],gpu[i]);
            return false;
        }
    }
    return true;
}
// 编译命令 nvcc ./softmaxWidthDot.cu -o main -lcurand
int main()
{
    const int N = 512;
    const int S = 256;
    float *mat_gpu, *mat_cpu;
    float *vec_gpu, *vec_cpu;
    cudaMalloc(&mat_gpu, sizeof(float) * N * S);
    cudaMalloc(&vec_gpu, sizeof(float) * S);
    mat_cpu = (float *)malloc(sizeof(float)*N*S);
    vec_cpu = (float *)malloc(sizeof(float)*S);
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, time(NULL));
    curandGenerateNormal(gen, mat_gpu, N * S, 0.5f, 1.0f);
    curandGenerateNormal(gen, vec_gpu, S, 0.5f, 1.0f);
    cudaMemcpy(mat_cpu,mat_gpu,sizeof(float)*N*S,cudaMemcpyDeviceToHost);
    cudaMemcpy(vec_cpu,vec_gpu,sizeof(float)*S,cudaMemcpyDeviceToHost);
  
    
    float *out_gpu;
    float *res;
    float *out_cpu;
    cudaMalloc(&out_gpu,sizeof(float)*N*S);
    out_cpu=(float *)malloc(sizeof(float)*N*S);
    res =  (float *)malloc(sizeof(float)*N*S); 
    softmax_fused_dot<256><<<512,256>>>(mat_gpu,vec_gpu,out_gpu,N,S);
    naive(mat_cpu,vec_cpu,out_cpu,N,S);
    cudaMemcpy(res,out_gpu,sizeof(float)*N*S,cudaMemcpyDeviceToHost);
    bool f =compare_res(out_cpu,res,N*S,1e-6f);
    if(f){
        printf("计算无误\n");
    }else{
        printf("计算有误\n");
    }
    curandDestroyGenerator(gen);
    cudaFree(mat_gpu);
    cudaFree(vec_gpu);
    cudaFree(out_gpu);
    free(mat_cpu);
    free(vec_cpu);
    free(res);
    free(out_cpu);
    return 0;
}