#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <random>
#include <chrono>
#include "./layernormal.h"
#define debug 0
inline float mean(float *x, int len)
{
    float sum = 0.0f;
    for (int i = 0; i < len; i++)
    {
        sum += x[i];
    }
    return sum / len;
}
inline float variance(float *x, float e, int len)
{
    float sum = 0.0f;
    for (int i = 0; i < len; i++)
    {
        sum += (x[i] - e) * (x[i] - e);
    }
    return sum / len;
}
void layer_normal_cpu(float *x, float *y, float gamma, const int N, const int K)
{
    const float eplison = 1e-5f;
    for (int i = 0; i < N; i++)
    {
        float m = mean(x + i * K, K);
        float v = variance(x + i * K, m, K);
#if debug
        if(i==0){
            printf("mean=%f\n",m);
            printf("variance=%f\n",1.0f/sqrt(v + eplison));
            printf("x=%f\n",x[0]);
        }   
#endif
        for (int j = 0; j < K; j++)
        {
            y[i * K + j] = gamma * (x[i * K + j] - m) / sqrt(v + eplison);
        }
    }
}
int main()
{
    const int N = 512;
    const int K = 256;
    constexpr int len = N * K;
    float *x = (float *)malloc(sizeof(float) * len);
    float *y = (float *)malloc(sizeof(float) * len);
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);

    // 2. 创建分布器
    std::uniform_real_distribution<float> distribution(0.0, 1.0); // [0.0, 1.0)
    for (int i = 0; i < len; i++)
    {
        x[i] = distribution(generator);
        y[i] = 0.0f;
    }
    layer_normal_cpu(x, y, 1.0f, N, K);
    float *x_gpu, *y_gpu;
    cudaMalloc(&x_gpu, sizeof(float) * len);
    cudaMalloc(&y_gpu, sizeof(float) * len);
    cudaMemcpy(x_gpu, x, sizeof(float) * len, cudaMemcpyHostToDevice);
    lanuch_kernel<256>(x_gpu, y_gpu, 1.0f, N, K);
    float *out = (float *)malloc(sizeof(float) * len);
    cudaMemcpy(out, y_gpu, sizeof(float) * len, cudaMemcpyDeviceToHost);
    bool f = true;
    for (int i = 0; i < len; i++)
    {
        if (std::abs(out[i] - y[i]) > 1e-5f)
        {
            f = false;
            printf("errors: index=%d diff=%f out=%f y=%f\n", i, std::abs(out[i] - y[i]), out[i], y[i]);
            break;
        }
    }
    if (f)
    {
        printf("计算无误\n");
    }
    else
    {
        printf("计算有误\n");
    }
    return 0;
}