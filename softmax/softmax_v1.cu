#include <stdio.h>
#include <stddef.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#define WARPSIZE 32


bool CheckResult(float *out, float* groudtruth, int N){
    for (int i = 0; i < N; i++){
      if(i == 0){
        printf("1st comparsion: %f and %f \n" , out[i], groudtruth[i] );
      }
      if (out[i] != groudtruth[i]) {
          return false;
      }
    }
    return true;
}
void softmaxCPU(float* input, float* result, int rows, int cols){
  for (int j = 0; j < rows; j++)
  {
    float total = 0;
    float MAX = 0;
    for(int i = 0; i < cols; i++)
    {
      MAX = max(input[j * cols + i], MAX);
    }
    for(int i = 0; i < cols; i++)
    {
      total += exp(input[j * cols + i] - MAX);
    }
    for(int i = 0; i < cols; i++)
    {
      result[j * cols + i] = exp(input[j * cols + i] - MAX) / total;
    }
  }
}
template <typename T, int VecSize>
struct alignas(sizeof(T)*VecSize) VectorType
{
  T val[VecSize];
};

template <int VecSize>
__device__ void load( float *dst, const float *src, int row, const int rowSize, const int col)
{
  using VecType = VectorType<float, VecSize>;
  const int offset = (row * rowSize + col) / VecSize;
   *reinterpret_cast<VecType*>(dst) = *(reinterpret_cast<VecType*>(const_cast<float*>(src)) + offset);
}
template <int VecSize>
__device__ void store(float *src,  float *dst, int row, const int rowSize, const int col)
{
  using VecType = VectorType<float, VecSize>;
  const int offest = (row * rowSize + col) / VecSize;
  *(reinterpret_cast<VecType *>(dst)+ offest) = *reinterpret_cast<VecType *>(src);
}
// 一个向量处理4个数据
// 一个线程处理8个向量
// 一个warp处理1024个数据
template <typename T>
struct MaxOp
{
  __device__ __forceinline__ T operator()(const T &a, const T &b)
  {
    return a > b ? a : b;
  }
};
template <typename T>
struct SumOp
{
  __device__ __forceinline__ T operator()(const T &a, const T &b)
  {
    return a + b;
  }
};
template <template <typename> class ReductionOp, typename T>
__forceinline__ __device__ T WarpReduce(T val)
{
  for (uint mask = WARPSIZE / 2; mask > 0; mask >>= 1)
  {
    val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}
template <typename T>
__inline__ __device__ T Exp(T val);

template <>
__inline__ __device__ float Exp<float>(float val)
{
  return __expf(val);
}
template <typename T>
__inline__ __device__ T Div(T a, T b);

template <>
__inline__ __device__ float Div<float>(float a, float b)
{
  return a / b;
}
template <typename T>
__inline__ __device__ T inf();
template <>
__inline__ __device__ float inf<float>()
{
  return 10000000000;
}
template <int packSize, int colsPerThreads, int warpWidth, int rowPerThreads>
__global__ void softmaxGPU(const float *src,  float *dst, const int rows, const int cols)
{
  float buf[rowPerThreads][colsPerThreads];
  constexpr int numPacks = colsPerThreads / packSize;
  const int globalWarpId = blockDim.y * blockIdx.y + threadIdx.y;
  const int numglobalWarp = gridDim.y * blockDim.y;
  const int laneid = threadIdx.x;
  const int step = numglobalWarp * rowPerThreads;
  for (int row = globalWarpId * rowPerThreads; row < rows; row += step)
  {
    float threadMax[rowPerThreads];
    for (int rowId = 0; rowId < rowPerThreads; rowId++)
    {
      threadMax[rowId] = -inf<float>();
      float *rowbuf = buf[rowId];
      // 8个数据
      for (int packId = 0; packId < numPacks; packId++)
      {
        const int col = (packId * warpWidth + laneid) * packSize;
        int offset = packId * packSize;
        if (col < cols)
        {

          load<packSize>(rowbuf+offset, src, row + rowId, cols, col );
          for (int i = 0; i < packSize; i++)
          {
            threadMax[rowId] = max(threadMax[rowId], rowbuf[offset + i]);
          }
        }
        else
        {
          for (int i = 0; i < packSize; i++)
            rowbuf[offset + i] = -inf<float>();
        }
      }
    }
    float warpMax[rowPerThreads];
    for (int rowId = 0; rowId < rowPerThreads; rowId++)
    {
      warpMax[rowId] =  WarpReduce<MaxOp, float>(threadMax[rowId]);
    }
    float threadSum[rowPerThreads];
    for(int rowId =0;rowId<rowPerThreads;rowId++)
    {
      threadSum[rowId] = 0;
      float *vecbuf = buf[rowId];
      for (int i = 0; i < colsPerThreads; i++)
      {
         vecbuf[i] = Exp<float>(vecbuf[i]-warpMax[rowId]);
         threadSum[rowId] +=vecbuf[i];
      }
      
    }
    float warpSum[rowPerThreads];
    for (int rowId = 0; rowId< rowPerThreads; rowId++)
    {
      warpSum[rowId] = WarpReduce<SumOp,float>(threadSum[rowId]);
    }
    for (int rowId = 0; rowId < rowPerThreads; rowId++)
    {
      float *vecbuf = buf[rowId];
      for (int i = 0; i < colsPerThreads; i++)
      {
        vecbuf[i] = Div<float>(vecbuf[i],warpSum[rowId]);
      }
      for (int i = 0; i < numPacks; i++)
      {
        const int col = (warpWidth*i+laneid)*packSize;
        int offest = i*packSize;
        if(col<cols){
          store<packSize>(vecbuf+offest,dst,row+rowId,cols,col);
        }
      }
    }
  }
}
int main(){
    float milliseconds = 0;
    const int N = 1000 * 1024;
    float *src = (float *)malloc(N * sizeof(float));
    float *d_src;
    cudaMalloc((void **)&d_src, N * sizeof(float));

    //int gridSize = ;//2d block, blockx=32,blocky=num warps in a block,griddimy=block nums
    //int blockSize = 256;
    float *dst = (float*)malloc(N * sizeof(float));
    float *d_dst;
    cudaMalloc((void **)&d_dst, N * sizeof(float));
    float *groudtruth = (float *)malloc(N * sizeof(float));

    for(int i = 0; i < N; i++){
        src[i] = 1;
    }

    softmaxCPU(src, groudtruth, 1000, 1024);

    cudaMemcpy(d_src, src, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(1, 125);//y轴125个block,
    dim3 Block(32, 8);//x轴32个threads组成一个warp访问一行,y轴8个threads,8*125=1000行
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    softmaxGPU<1, 1024 / 32, 32, 1><<<Grid, Block>>>(d_src, d_dst, 1000, 1024);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(dst, d_dst, N * sizeof(float), cudaMemcpyDeviceToHost);
    bool is_right = CheckResult(dst, groudtruth, N);
    if(is_right) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
        for(int i=0;i<10;i++){
            printf("%lf ",dst[i]);
        }
        printf("\n");
    }
    printf("WarpSoftmax latency = %f ms\n", milliseconds);

    cudaFree(d_src);
    cudaFree(d_dst);
    free(src);
    free(dst);
    free(groudtruth);
}