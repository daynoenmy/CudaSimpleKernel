#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <curand.h>
#include <cuda_runtime.h>
//
#define SQRT_2_PI M_SQRT2 * M_2_SQRTPI * 0.5f
__inline__ __device__ float gelu_tanh_approximate(float x)
{
    return 0.5f * x * (1.0f + tanhf(SQRT_2_PI * (x + 0.044715F * x * x * x)));
}
float gelu_tanh_approximate_cpu(float x)
{
    return 0.5f * x * (1.0f + tanhf(SQRT_2_PI * (x + 0.044715F * x * x * x)));
}
__global__ void gelu_baseline(float *input, float *out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = gelu_tanh_approximate(input[idx]);
}
template <const int size>
void gelu_cpu(float *input, float *out)
{
    for (int i = 0; i < size; i++)
    {
        out[i] = gelu_tanh_approximate_cpu(input[i]);
    }
}
bool compare(float *ref,float *out,const int len,float dif)
{
    for (int i = 0; i < len; i++)
    {
        if(abs(ref[i]-out[i])>dif){
            printf("index=%d,ref:%f,out:%f\n",i,ref[i],out[i]);
            return false;
        }
    }
    return true;
    
}

int main()
{
    float *data, *out;
    const int batch_size = 512;
    const int seq_len = 1024;
    constexpr int len = batch_size * seq_len;
    cudaMalloc(&data, sizeof(float) * len);
    cudaMalloc(&out, sizeof(float) * len);
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    curandGenerateUniform(gen, data, len);
    curandDestroyGenerator(gen);
    float *input = (float *)malloc(sizeof(float) * len);
    float *ref = (float *)malloc(sizeof(float) * len);
    cudaMemcpy(input, data, sizeof(float) * len, cudaMemcpyDeviceToHost);
    printf("CPU开始计算\n");
    gelu_cpu<len>(input, ref);
    printf("CPU计算完成\n");
    float *res = (float *)malloc(sizeof(float)*len);
    const int block_size = 256;
    dim3 block(block_size);
    dim3 grid(len/block_size);
    gelu_baseline<<<grid,block>>>(data,out);
    cudaMemcpy(res,out,sizeof(float)*len,cudaMemcpyDeviceToHost);
    bool f = compare(ref,res,len,1e-5);
    if(f){
        printf("计算无误\n");
    }else{
        printf("计算有误\n");
    }
}