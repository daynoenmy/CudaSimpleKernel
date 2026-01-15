#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
// Q(B,Nh,N,d)
__global__ void forward_kernel(const float *Q, const float *K, const float *V,
                               const int N, const int d, const int Tc,
                               const int Tr, const int Bc, const int Br,
                               const float softmax_scale, float *l, float *m, float *O)
{
    int tx = threadIdx.x;
    int bx = blockIdx.x; // batch
    int by = blockIdx.y; // head
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);
    extern __shared__ float sram[];
    int tile_size = Bc * d;
    float *Qi = sram;
    float *Kj = &sram[tile_size];
    float *Vj = &sram[tile_size * 2];
    float *S = &sram[tile_size * 3];
    for (int j = 0; j < Tc; j++)
    {
        for (int x = 0; x < d; x++)
        {
            Kj[(tx*d)+x] = K[qkv_offset + (tile_size*j)+(tx*d)+x];
            Vj[(tx*d)+x] = V[qkv_offset +(tile_size*j)+(tx*d)+x];            
        }
        __syncthreads();
        for (int i = 0; i < Tr; i++)
        {
            
        }
        
    }
    

}

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V)
{
    const int Bc = 32;
    const int Br = 32;
    // batch_size
    const int B = Q.size(0);
    // n_head
    const int nh = Q.size(1);
    // seq_len
    const int N = Q.size(2);
    // head_emdb
    const int d = Q.size(3);
    // 初始化阶段
    const int Tc = ceil((float)N / Bc);
    const int Tr = ceil((float)N / Br);
    const float softmax_scale = rsqrtf(d);

    auto O = torch::zeros_like(Q);
    auto l = torch::zeros({B, nh, N});
    auto m = torch::full({B, nh, N}, -INFINITY);

    torch::Device device(torch::kCUDA);
    const int sram_size = (3 * Bc * d * sizeof(float)) + Bc * Br * sizeof(float);
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d\n", max_sram_size, sram_size);
    l = l.to(device);
    m = m.to(device);
    dim3 grid_dim(B, nh);
    // (B,nh)块
    // 一块（32）
    dim3 block_dim(Bc);
}