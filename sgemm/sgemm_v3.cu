#include <cstdio>
void initMat(const int row, const int col, float *a)
{
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            a[i * col + j] = 2.0f * (float)drand48() - 1.0f;
        }
    }
}

void gemmCpu(float *a, float *b, float *c, const int M, const int K, const int N)
{
    for (int m = 0; m < M; m++)
    {
        for (int n = 0; n < N; n++)
        {
            float temp = 0.0f;
            for (int k = 0; k < K; k++)
            {
                temp += a[m * K + k] * b[k * N + n];
            }
            c[m * N + n] = temp;
        }
    }
}
bool compareMat(float *mat_A, float *mat_B, const int rows, const int cols)
{
    float max_diff = 0.0f;
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            float a = mat_A[i * cols + j];
            float b = mat_B[i * cols + j];
            float diff = fabsf(a - b);
            if (diff > max_diff)
            {
                max_diff = diff;
            }
            if (diff > 0.5f)
            {
                printf("Error at (%d, %d): diff = %f, a = %f, b = %f\n", i, j, diff, a, b);
                return false;
            }
        }
    }
    return true;
}
template <int Block_Size, int Stride>
__global__ void sgemm_v3(float *mat_A, float *mat_B, float *mat_C, int M, int K, int N)
{

    constexpr int STEP = Block_Size * Stride;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float *mat_A_start_ptr = mat_A + blockIdx.y * STEP * K;
    float *mat_B_start_ptr = mat_B + blockIdx.x * STEP;
    __shared__ float mem_A[STEP][STEP];
    __shared__ float mem_B[STEP][STEP];
    float temp[Stride][Stride] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int k = 0; k < K; k += STEP)
    {
        for (int i = 0; i < Stride; i++)
        {
            for (int j = 0; j < Stride; j++)
            {
                int a_row = ty + i * Block_Size;
                int a_col = tx + j * Block_Size + k;
                int b_row = ty + i * Block_Size + k;
                int b_col = tx + j * Block_Size;
                if (a_row < M && a_col < K)
                    mem_A[ty + i * Block_Size][tx + j * Block_Size] = mat_A_start_ptr[(ty + i * Block_Size) * K + tx + j * Block_Size + k];
                else
                    mem_A[ty + i * Block_Size][tx + j * Block_Size] = 0.0f;
                if (b_row < K && b_col < N)
                    mem_B[ty + i * Block_Size][tx + j * Block_Size] = mat_B_start_ptr[tx + j * Block_Size + (ty + i * Block_Size + k) * N];
                else
                    mem_B[ty + i * Block_Size][tx + j * Block_Size] = 0.0f;
            }
        }
        __syncthreads();
        for (int i = 0; i < Stride; i++)
        {
            for (int j = 0; j < Stride; j++)
            {
                for (int k = 0; k < STEP; k++)
                {
                    temp[i][j] += mem_A[ty + i * Block_Size][k] * mem_B[k][tx + j * Block_Size];
                }
            }
        }
        __syncthreads();
    }
    float *mat_C_start_ptr = mat_C + N * blockIdx.y * STEP + blockIdx.x * STEP;
    for (int i = 0; i < Stride; i++)
    {
        for (int j = 0; j < Stride; j++)
        {
            mat_C_start_ptr[N * (ty + i * Block_Size) + tx + j * Block_Size] = temp[i][j];
        }
    }
}
int main()
{
    const int m = 512;
    const int n = 256;
    const int k = 512;
    constexpr size_t mem_size_A = m * k * sizeof(float);
    constexpr size_t mem_size_B = k * n * sizeof(float);
    constexpr size_t mem_size_C = m * n * sizeof(float);
    // 一个block最多1024个线程
    const int blockSize = 16;
    const int stride = 2;
   
    dim3 Block(blockSize, blockSize);
    dim3 Grid((n + blockSize - 1) / blockSize / stride,(m+blockSize-1)/blockSize/stride);
    float *mat_A = (float *)malloc(mem_size_A);
    float *mat_B = (float *)malloc(mem_size_B);
    float *mat_C = (float *)malloc(mem_size_C);
    initMat(m, k, mat_A);
    initMat(k, n, mat_B);
    gemmCpu(mat_A, mat_B, mat_C, m, k, n);
    float *mat_A_gpu;
    float *mat_B_gpu;
    float *mat_C_gpu;
    float *mat_res = (float *)malloc(mem_size_C);

    cudaMalloc(&mat_C_gpu, mem_size_C);
    cudaMalloc(&mat_A_gpu, mem_size_A);
    cudaMalloc(&mat_B_gpu, mem_size_B);
    cudaMemcpy(mat_A_gpu, mat_A, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(mat_B_gpu, mat_B, mem_size_B, cudaMemcpyHostToDevice);
    sgemm_v3<blockSize, stride><<<Grid, Block>>>(mat_A_gpu, mat_B_gpu, mat_C_gpu, m, k, n);
    cudaMemcpy(mat_res, mat_C_gpu, mem_size_C, cudaMemcpyDeviceToHost);
    if (compareMat(mat_C, mat_res, m, n))
    {
        printf("计算成功\n");
    }
    else
    {
        printf("计算不成功\n");
    }
    free(mat_A);
    free(mat_B);
    free(mat_C);
    free(mat_res);
    cudaFree(mat_A_gpu);
    cudaFree(mat_B_gpu);
    cudaFree(mat_C_gpu);

    return 0;
}
