#include <cstdio>
#include <cuda_runtime.h>
#include "./rmsnormal.h"
#include <cstdlib>
#include <random>
#include <chrono>

void init_array(float *x, const int len)
{
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);

    // 2. 创建分布器
    std::uniform_real_distribution<float> distribution(0.0, 1.0); // [0.0, 1.0)
    for (int i = 0; i < len; i++)
    {
        x[i] = distribution(generator);
    }
}

void rmsnormal_cpu(float *x, float *y, float gamma, const int N, const int K)
{
    const float EPSILON = 1e-5f;
    for (int i = 0; i < N; i++)
    {
        float sum = 0.0f;
        int index = 0;
        for (int j = 0; j < K; j++)
        {
            index = i * K + j;
            sum += (x[index] * x[index]);
        }
        float rms = sqrt(sum / K);
        for (int j = 0; j < K; j++)
        {
            index = i * K + j;
            y[index] = gamma * x[index] / (rms + EPSILON);
        }
    }
}
int main()
{
    const int K = 256;
    const int N = 512;
    float *x = (float *)malloc(sizeof(float) * N * K);
    float *y = (float *)malloc(sizeof(float) * N * K);
    init_array(x, N * K);
    rmsnormal_cpu(x, y, 1.0f, N, K);
    float *x_gpu, *y_gpu;
    cudaMalloc(&x_gpu, sizeof(float) * K * N);
    cudaMemcpy(x_gpu, x, sizeof(float) * K * N, cudaMemcpyHostToDevice);
    cudaMalloc(&y_gpu, sizeof(float) * K * N);
    launch_kernel<64>(x_gpu, y_gpu, 1.0f, N, K);
    float *out = (float *)malloc(sizeof(float) * N * K);
    cudaMemcpy(out, y_gpu, sizeof(float) * N * K, cudaMemcpyDeviceToHost);
    bool f = true;
    for (int i = 0; i < N * K; i++)
    {
        if (std::abs(out[i] - y[i])>1e-5)
        {
            f = false;
            printf("计算有误\n");
            printf("index = %d\n out=%f,y=%f\n dif=%f\n", i,out[i],y[i],std::abs(out[i] - y[i]));
            break;
        }
    }
    if (f)
    {
        printf("计算无误\n");
    }
    free(x);
    free(y);
    free(out);
    cudaFree(x_gpu);
    cudaFree(y_gpu);
}