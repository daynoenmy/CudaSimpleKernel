#include<cstdio>
void initMat(const int row,const int col,float *a)
{
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            a[i*col+j] = 1.0f; //2.0f*(float)drand48()-1.0f;
        }
        
    }
    
}


void gemmCpu(float *a,float *b,float *c,const int M,const int K,const int N)
{
  for(int m=0;m<M;m++)
  {
    for(int n=0;n<N;n++)
    {
      float temp = 0.0f;
      for(int k=0;k<K;k++)
      {
        temp += a[m*K+k]*b[k*K+n];
      
      }
      c[m*N+n] = temp;
    }
  }
}
__global__ void gemmGPU(float *mat_A,float *mat_B,float *mat_C,const int M,const int K,const int N)
{
 
  const int col = threadIdx.x + blockIdx.x*blockDim.x;
  const int row = threadIdx.y+ blockIdx.y*blockDim.y; 
  float *mat_A_start_ptr = mat_A + blockIdx.y*blockDim.y*K;
  float *mat_B_start_ptr = mat_B + blockIdx.x*blockDim.x;
  float temp =0.0f;
  
  for (int i =0; i <K ; i++)
  {
   
    temp += mat_A_start_ptr[threadIdx.y*K+i]*mat_B_start_ptr[i*N+threadIdx.x];
  }

  mat_C[col+row*N] = temp;
}
bool compareMat(float *mat_A,float *mat_B,const int row,const int col)
{
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            if(abs(mat_A[i*col+j]-mat_B[i*col+j])>1e-6)
            {
                printf("error a = %f b=%f",mat_A[i*col+j],mat_B[i*col+j]);
                printf("error diff:%f rows=%d,col=%d\n",abs(mat_A[i*col+j]-mat_B[i*col+j]),i,j);
                return false;
            }
        }
        
    }
    return true; 
}

int main()
{
    const int m=512;
    const int n=256;
    const int k =128;
    constexpr size_t mem_size_A = m*k*sizeof(float);
    constexpr size_t mem_size_B = k*n*sizeof(float);
    constexpr size_t mem_size_C = m*n*sizeof(float);
    // 一个block最多1024个线程
    const int blockSize = 16;
    const int gridSize = (m+blockSize-1)/blockSize;
    dim3 Block(blockSize,blockSize);
    dim3 Grid((n+blockSize-1)/blockSize,(m+blockSize-1)/blockSize);
    float *mat_A = (float *)malloc(mem_size_A);
    float *mat_B = (float *)malloc(mem_size_B);
    float *mat_C = (float *)malloc(mem_size_C);
    initMat(m,k,mat_A);
    initMat(k,n,mat_B);
    gemmCpu(mat_A,mat_B,mat_C,m,k,n);
    float *mat_A_gpu;
    float *mat_B_gpu;
    float *mat_C_gpu;
    float *mat_res = (float *)malloc(mem_size_C);
   
    cudaMalloc(&mat_C_gpu,mem_size_C);
    cudaMalloc(&mat_A_gpu,mem_size_A);
    cudaMalloc(&mat_B_gpu,mem_size_B);
    cudaMemcpy(mat_A_gpu,mat_A,mem_size_A,cudaMemcpyHostToDevice);
    cudaMemcpy(mat_B_gpu,mat_B,mem_size_B,cudaMemcpyHostToDevice);
    gemmGPU<<<Grid,Block>>>(mat_A_gpu,mat_B_gpu,mat_C_gpu,m,k,n);
    cudaMemcpy(mat_res,mat_C_gpu,mem_size_C,cudaMemcpyDeviceToHost);
    if(compareMat(mat_C,mat_res,m,n))
    {
      printf("计算成功\n");
    }else
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
