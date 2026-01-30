#include<cstdio>
#include<cuda.h>
#include<cuda_runtime.h>
#include<assert.h>
#include<curand.h>
#include<cmath>
// input [seq_len,N]
void rope_cpu(float *input,float *out,int seq_len,int head_dim){
    assert(head_dim%2==0);
 
    int dim = head_dim/2;
    for (int i = 0; i < seq_len; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            float token_1 = input[i*head_dim+j*2];
            float token_2 = input[i*head_dim+j*2+1];
            float theta = i/powf(10000.0f,float(2*j)/(head_dim*1.0f));
            out[i*head_dim+j*2] = cosf(theta)*token_1 - sinf(theta)*token_2;
            out[i*head_dim+j*2+1] = sinf(theta)*token_1 + cosf(theta)*token_2;
        }   
    }
   
}
// 256*256 % 256
__global__ void rope_kernel_baseline(float *input,float *output,int seq_len,int head_dim)
{
    const float d = 10000.0f;
    const int dim  = head_dim/2;
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    float x1 = input[idx*2];
    float x2 = input[idx*2+1];
    int token_pos = idx/dim;
    int token_idx = idx % dim;
    float exp_v = 1.0f / powf(d,2*token_idx/(head_dim*1.0f));
    float sin_v = sinf(token_pos*exp_v);  
    float cos_v = cosf(token_pos*exp_v);
    float out1 = x1*cos_v-sin_v*x2;
    float out2 = x1*sin_v+cos_v*x2;
    output[idx*2] = out1;
    output[idx*2+1] = out2;
}
bool dif(float x1,float x2,float d){
    if(fabsf(x1-x2)>d){
        return false;
    }
    return true;
}
int main()
{
    const int seq_len = 256;
    const int head_dim = 512;
    constexpr int len = seq_len*head_dim;
    float *data,*g_res;
    float *input=(float *)malloc(sizeof(float)*len);
    float *output = (float *)malloc(sizeof(float)*len);
    cudaMalloc(&data,sizeof(float)*seq_len*head_dim);
    cudaMalloc(&g_res,sizeof(float)*len);
    curandGenerator_t gen;
    curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen,123124);
    curandGenerateUniform(gen,data,len);
    cudaMemcpy(input,data,sizeof(float)*len,cudaMemcpyDeviceToHost);
    rope_cpu(input,output,seq_len,head_dim);
    dim3 block(256);
    dim3 grid(256);
    rope_kernel_baseline<<<grid,block>>>(data,g_res,seq_len,head_dim);
    float *res =(float *)malloc(sizeof(float)*len);
    cudaMemcpy(res,g_res,sizeof(float)*len,cudaMemcpyDeviceToHost);
    bool f = false;
    for (int i = 0; i < len; i++)
    {
        if(!dif(res[i],output[i],1e-4)){
            printf("index =%d res = %f output = %f\n",i,res[i],output[i]);
            f = true;
        }
    }
    if(f){
        printf("计算有误\n");
    }else{
        printf("计算无误\n");
    }
    
    curandDestroyGenerator(gen);
    cudaFree(data);   
}