#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
// 定义数据
#define  N 10000000
#define BLOCKSIZE 256
#define GRIDSIZE 32
#define TOPK 20
// CPU 与GPU 可同时访问
__managed__ int sources[N];
__managed__ int res[TOPK];
__managed__ int blockRes[TOPK*GRIDSIZE];

//向列表插入数据 
// arr 列表
// value 插入值
// len 列表长度
__device__ __host__ void insertValue(int *arr,int value,int len){
    if(value<arr[len -1]){
        return ;
    }
    for(int i =len-2;i>=0;i--){
        if(value>arr[i]){
            arr[i+1] = arr[i];
        }else{
            arr[i+1] = value;
            return ;
        }
    }
    arr[0] = value;
}   
void topKCPU(int *in,int *out,int len){
    for (uint i = 0; i < TOPK; i++)
    {
        out[i]= INT_MIN; 
    }
    for (uint i = 0; i < len; i++)
    {
        insertValue(out,in[i],TOPK);
    }
    return;
    
    
}
__global__ void topkGPU(int *in,int *out,int len){
    int topArray[TOPK];
    for (uint i = 0; i < TOPK; i++)
    {
        topArray[i] = INT_MIN;
    }
    
    uint totalNums = blockDim.x*gridDim.x;
    uint localx  = threadIdx.x;
    uint globalx = localx + blockDim.x*blockIdx.x;
    __shared__ int smem[TOPK*BLOCKSIZE];
    for (uint idx = globalx; idx < len; idx+=totalNums)
    {
        insertValue(topArray,in[idx],TOPK);
    }
    for (uint i = 0; i < TOPK; i++)
    {
        smem[localx*TOPK+i] = topArray[i];
    }
    __syncthreads();
    for (int index = BLOCKSIZE/2; index>0;index>>=1)
    {
        if(localx<index){
            for (uint i = 0; i < TOPK; i++)
            {
                insertValue(topArray,smem[(index+localx)*TOPK+i],TOPK);
            }
            
        }
        __syncthreads();
        if(localx<index){
            for (uint i = 0; i < TOPK; i++)
            {
                smem[localx*TOPK+i] = topArray[i];
            }
            
        }
        __syncthreads();
    }
    if(localx==0){
        for (uint i = 0; i < TOPK; i++)
        {
            out[blockIdx.x*TOPK+i]= smem[i];
        }
        
    }

}
int main()
{
   
    printf("初始化\n");
    for (uint i = 0; i < N; i++)
    {
        sources[i] = rand();
    }
    cudaEvent_t start,stop;
    float milliseconds=0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    dim3 block(BLOCKSIZE);
    dim3 grid(GRIDSIZE);
    printf("GUP计算\n");
    topkGPU<<<grid,block>>>(sources,blockRes,N);
    // 不是GRIDSIZE 一块有TOPK个数据 所以总数是块数*TOPK
    topkGPU<<<1,block>>>(blockRes,res,GRIDSIZE*TOPK);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds,start,stop);
    printf("GPU计算完成 time = %.2f\n",milliseconds);
    int *out = new int[TOPK];
    topKCPU(sources,out,N);
    for (uint i = 0; i < TOPK; i++)
    {
        if(out[i]!=res[i]){
            
            printf("计算有误 index=%d,CPU=%d,GPU=%d\n",i,out[i],res[i]);
            delete out;
            return 0;
        }
        printf("index=%d,CPU=%d,GPU=%d\n",i,out[i],res[i]);
    }
    printf("计算无误\n");
    delete out;
    return 0;
}