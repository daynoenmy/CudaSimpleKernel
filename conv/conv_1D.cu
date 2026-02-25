#include <cstdio>

// [1,2,3,4,5,6,7]
// [1,1,1,1,1]
template <const int in_len, const int mask_len>
void conv_1D_cpu(float *in, float *out, float *mask)
{
  constexpr int radius = mask_len / 2;
  for (int i = 0; i < in_len; i++)
  {
    float sum = 0.0f;
    for (int j = 0; j < mask_len; j++)
    {
      int idx = i + (j - radius);
      if (idx >= 0 && idx < in_len)
      {
        sum += in[idx] * mask[j];
      }
    }
     out[i] = sum;
  }
}
template<const int in_len,const int mask_len>
__global__ void conv1d_baseline(float *in,float *out,float *mask)
{
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  float sum = 0.0f;
  constexpr int radius =  mask_len/2;
  for (int i = 0; i < mask_len i++)
  {
      int j = idx + (i - radius);
      if(j>=0&&j<in_len){
        sum+= in[j]*mask[i];
      }
  }
  out[idx] = sum;
}
bool compare(float *out, float *ref, float dif, const int len)
{
  for (int i = 0; i < len; i++)
  {
    if (abs(out[i] - ref[i]) > dif)
    {
      printf("err:index=%d,ref=%f,out=%f\n", i, out[i], ref[i]);
      return false;
    }
  }
  return true;
}
// 验证 conv
void TestCase()
{
  const int in_len = 7;
  const int mask_len = 5;
  {
    float in[in_len] = {1, 2, 3, 4, 5, 6, 7};
    float mask[mask_len] = {1, 1, 1, 1, 1};
    float ref[in_len] = {6, 10, 15, 20, 25, 22, 18};
    float out[in_len];
    conv_1D_cpu<in_len, mask_len>(in, out, mask);
    bool f = compare(out, ref, 1e-6, in_len);
    if (f)
    {
      printf("计算无误\n");
    }
    else
    {
      printf("计算有误\n");
    }
  }
  {
    float in[in_len] = {1, 2, 3, 4, 5, 6, 7};
    float mask[mask_len] = {1, 2, 3, 4, 5};
    float ref[in_len] = {26, 40, 55, 70, 85, 60, 38};
    float out[in_len];
    conv_1D_cpu<in_len, mask_len>(in, out, mask);
    bool f = compare(out, ref, 1e-6, in_len);
    if (f)
    {
      printf("计算无误\n");
    }2
    else
    {
      printf("计算有误\n");
    }
  }
}
int main()
{
  float *a,*b;
  
  return 0;
}
