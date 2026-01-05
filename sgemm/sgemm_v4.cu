#include <cstdio>
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
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
template <const int M_NUM_PER_BLOCK, const int N_NUM_PER_BLOCK, const int K_NUM_PER_BLOCK, const int NUM_PER_THREADS>
__global__ void sgemm_v4(float *mat_A, float *mat_B, float *mat_C, const int M, const int K, const int N)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float *mat_A_ptr_start = mat_A + blockIdx.y * M_NUM_PER_BLOCK * K;
    float *mat_B_ptr_start = mat_B + blockIdx.x * N_NUM_PER_BLOCK;
    __shared__ float memA[M_NUM_PER_BLOCK][K_NUM_PER_BLOCK];
    __shared__ float memB[K_NUM_PER_BLOCK][N_NUM_PER_BLOCK];
    float temp[NUM_PER_THREADS]={0.0f}; 
    for (int s = 0; s < K; s += K_NUM_PER_BLOCK)
    {
        FETCH_FLOAT4(memA[ty][tx*NUM_PER_THREADS]) = FETCH_FLOAT4(mat_A_ptr_start[K*ty+tx*NUM_PER_THREADS+s]);
        FETCH_FLOAT4(memB[ty][tx*NUM_PER_THREADS]) = FETCH_FLOAT4(mat_B_ptr_start[N*(ty+s)+tx*NUM_PER_THREADS]);
        __syncthreads();
        for (int i = 0; i < NUM_PER_THREADS; i++)
        {
            for (int k = 0; k < K_NUM_PER_BLOCK; k++)
            {
                temp[i] += memA[ty][k]*memB[k][tx*NUM_PER_THREADS+i];
            }
            
        }
        __syncthreads();
        
    }
    float *mat_C_ptr_start= mat_C + blockIdx.y*M_NUM_PER_BLOCK*N+blockIdx.x*N_NUM_PER_BLOCK;
    for (int i = 0; i <NUM_PER_THREADS; i++)
    {
        mat_C_ptr_start[ty*N+tx*NUM_PER_THREADS+i]=temp[i];
    }
    

}
int main()
{
    const int m = 512;
    const int n = 256;
    const int k = 128;
    constexpr size_t mem_size_A = m * k * sizeof(float);
    constexpr size_t mem_size_B = k * n * sizeof(float);
    constexpr size_t mem_size_C = m * n * sizeof(float);
    // 一个block最多1024个线程
    const int blockSize = 32;
   
    dim3 Block(blockSize/4, blockSize);
    dim3 Grid(n/blockSize,m/blockSize);
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
    const int M_NUM_PER_BLOCK =32;
    const int N_NUM_PER_BLOCK =32;
    const int K_NUM_PER_BLOCK =32;
    const int NUM_PER_THREADS =4;
    
    cudaMalloc(&mat_C_gpu, mem_size_C);
    cudaMalloc(&mat_A_gpu, mem_size_A);
    cudaMalloc(&mat_B_gpu, mem_size_B);
    cudaMemcpy(mat_A_gpu, mat_A, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(mat_B_gpu, mat_B, mem_size_B, cudaMemcpyHostToDevice);
    sgemm_v4<M_NUM_PER_BLOCK,N_NUM_PER_BLOCK,K_NUM_PER_BLOCK,NUM_PER_THREADS><<<Grid,Block>>>(mat_A_gpu,mat_B_gpu,mat_C_gpu,m,k,n);
    // sgemm_v3<blockSize, stride><<<Grid, Block>>>(mat_A_gpu, mat_B_gpu, mat_C_gpu, m, k, n);
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