#include <cstdio>

// 只在一个 .cpp 里定义一次
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define DIM 1024
#define EPOCH 20
struct cuComplex
{
    /* data */
    float r;
    float i;
    __device__ cuComplex(float a, float b) : r(a), i(b) {}
    __device__ inline float magnitude2() const { return r * r + i * i; }
    __device__ cuComplex operator*(const cuComplex &a)
    {
        return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
    }
    __device__ cuComplex operator+(const cuComplex &a)
    {
        return cuComplex(r + a.r, i + a.i);
    }
};
__device__ int julia(int x, int y)
{
    const float scale = 1.5f;
    float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
    float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

    cuComplex c(0.188, 0.78603);
    cuComplex a(jx, jy);
    for (int i = 0; i < EPOCH; i++)
    {
        a = a * a + c;
        if (a.magnitude2() > 1000.0f)
            return 0;
    }
    return 1;
}

__global__ void draw_kernel(unsigned char *m)
{
    int x = blockIdx.x;
    int y = blockIdx.y;
    int idx = x + y * DIM;
    int value = julia(x, y);
    m[idx] = 255*value;
}

int main()
{
    unsigned char img[DIM * DIM * 1];
    unsigned char *img_gpu;
    cudaMalloc(&img_gpu,sizeof(unsigned char)*DIM*DIM);
    dim3 grid(DIM, DIM);
    draw_kernel<<<grid, 1>>>(img_gpu);
    cudaMemcpy(img,img_gpu,sizeof(unsigned char)*DIM*DIM,cudaMemcpyDeviceToHost);
    if (!stbi_write_bmp("julia.bmp", DIM, DIM, 1, img))
    {
        std::printf("write bmp failed\n");
        return 1;
    }
    std::printf("write image success\n");
}