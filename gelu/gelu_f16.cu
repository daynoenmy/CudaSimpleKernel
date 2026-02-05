#include <cstdio>
#include <cuda.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <random>
#define HALF_1 __float2half(1.0f)
#define HALF_2 __float2half(2.0f)
#define HALF_DIV2 __float2half(0.5f)
#define HALF_SQRT_2_PI \
    __float2half(M_SQRT2) * __float2half(M_2_SQRTPI) * HALF_DIV2
#define FLOAT_SQRT_2_PI M_SQRT2 * M_2_SQRTPI * 0.5f
#define HALF_V_APP __float2half(0.044715f)
#define FLOAT_V_APP 0.044715f
__device__ __forceinline__ half gelu_tanh_approximate(half x)
{
    half cube_x = x * x * x;
    half inner = HALF_SQRT_2_PI * (x + HALF_V_APP * cube_x);
    return HALF_DIV2 * x * (HALF_1 + (hexp(inner * HALF_2) - HALF_1) / (hexp(inner * HALF_2) + HALF_1));
}
half gelu_tanh_approximate_cpu(float x)
{
    float cube_x = x * x * x;
    float inner = FLOAT_SQRT_2_PI * (x + FLOAT_V_APP * cube_x);
    return __float2half(0.5f * x *(1+tanhf(inner)));
}
__global__ void gelu_f16(half *in, half *out)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = gelu_tanh_approximate(in[idx]);
}
__global__ void gelu_f16x8_unpack(half *in,half *out)
{
    int idx = (blockIdx.x*blockDim.x + threadIdx.x)*8;
    half2 reg_x_0 = (reinterpret_cast<half2 *>(&in[idx+0]))[0];
    half2 reg_x_1 = (reinterpret_cast<half2 *>(&in[idx+2]))[0];
    half2 reg_x_2 = (reinterpret_cast<half2 *>(&in[idx+4]))[0];
    half2 reg_x_3 = (reinterpret_cast<half2 *>(&in[idx+6]))[0];
    half2 reg_y_0,reg_y_1,reg_y_2,reg_y_3;
    reg_y_0.x = gelu_tanh_approximate(reg_x_0.x);
    reg_y_0.y = gelu_tanh_approximate(reg_x_0.y);
    reg_y_1.x = gelu_tanh_approximate(reg_x_1.x);
    reg_y_1.y = gelu_tanh_approximate(reg_x_1.y);
    reg_y_2.x = gelu_tanh_approximate(reg_x_2.x);
    reg_y_2.y = gelu_tanh_approximate(reg_y_2.y);
    reg_y_3.x = gelu_tanh_approximate(reg_x_3.x);
    reg_y_3.y = gelu_tanh_approximate(reg_x_3.y);
    (reinterpret_cast<half2 *>(&out[idx+0]))[0] = reg_y_0;
    (reinterpret_cast<half2 *>(&out[idx+2]))[0] = reg_y_1;
    (reinterpret_cast<half2 *>(&out[idx+4]))[0] = reg_y_2;
    (reinterpret_cast<half2 *>(&out[idx+6]))[0] = reg_y_3;
}
__global__ void gelu_f16x8(half *in,half *out)
{
    int idx = (blockIdx.x*blockDim.x+threadIdx.x)*8;
    half pack_x[8],pack_y[8];
    (reinterpret_cast<float4 *>(&pack_x[0]))[0] =(reinterpret_cast<float4 *>(&in[idx]))[0];
    #pragma unroll
    for (int i = 0; i < 8; i++)
    {
        pack_y[i] = gelu_tanh_approximate(pack_x[i]);
    }
    (reinterpret_cast<float4 *>(&out[idx]))[0] = (reinterpret_cast<float4 *>(&pack_y[0]))[0];
}
template <const int len>
void gelu_cpu(half *in, half *out)
{
    for (int i = 0; i < len; i++)
    {
        out[i] = gelu_tanh_approximate_cpu(__half2float(in[i]));
    }
}
int main()
{
    half *in;
    half *out;
    const int batch_size = 512;
    const int seq_len = 256;
    constexpr int len = batch_size * seq_len;
    in = (half *)malloc(sizeof(half) * len);
    out = (half *)malloc(sizeof(half) * len);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < len; i++)
    {
        in[i] = __float2half(dist(gen));
    }
    gelu_cpu<len>(in, out);
    half *gin, *gout;
    cudaMalloc(&gin, sizeof(half) * len);
    cudaMalloc(&gout, sizeof(half) * len);
    cudaMemcpy(gin, in, sizeof(half) * len, cudaMemcpyHostToDevice);
    dim3 block(256/8);
    dim3 grid(batch_size);
    gelu_f16x8<<<grid, block>>>(gin, gout);
    half *res = (half *)malloc(sizeof(half) * len);
    cudaMemcpy(res, gout, sizeof(half) * len, cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < len; i++)
    {
        float a = __half2float(res[i]);
        float b = __half2float(out[i]);
        if (abs(a - b) > 1e-3)
        {
            printf("计算有误\n");
            printf("index=%ld,ref=%f,out=%f\n", i, b, a);
            free(in);
            free(out);
            free(res);
            cudaFree(gin);
            cudaFree(gout);
            return 0;
        }
    }
    printf("计算无误\n");
    free(in);
    free(out);
    free(res);
    cudaFree(gin);
    cudaFree(gout);
    return 0;
}