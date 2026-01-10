#include <cstdio>
// 外积替换内积实现矩阵乘法
template<const int M_NUM_PER_BLOCK,const int N_NUM_PER_BLOCK,const int K_NUM_PER_BLOCK,const int NUM_PER_THREAD>
__global__ void sgemm_v5(float *mat_A,float *mat_B,float *mat_C,const int M,const int K,const int N)
{
    uint ty = threadIdx.y;
    uint tx = threadIdx.x;
    float *mat_A_start_ptr = 
    float *mat_B_start_ptr = 
    __shared__ float memA[M_NUM_PER_BLOCK][K_NUM_PER_BLOCK];
    __shared__ float memB[K_NUM_PER_BLOCK][N_NUM_PER_BLOCK];
    float temp[NUM_PER_THREAD];
    for (int s = 0; s <K  ; s+=K_NUM_PER_BLOCK)
    {
        
    }
    
    

}
int main()
{
    int m=1024;
    int k = 1024;
    int n=1024;
    const int blockSize = 256;
}