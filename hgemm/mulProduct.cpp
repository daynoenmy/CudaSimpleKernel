#include<cstdio>
#include<vector>
using std::vector;
void mul(vector<float>& mat_x,vector<float>& mat_y,vector<float>& out,int M,int N,int K)
{
    for (int i = 0; i < K; i++)
    {
        for (int x = 0; x < M; x++)
        {
            for (int y = 0; y < N; y++)
            {
                int index =   x*N+y;
                out[index]+= mat_x[x*K+i]*mat_y[y+i*K];                
            }
            
        }
        
    }
    
}
int main()
{
    vector<float> mat_x(4);
    vector<float> mat_y(4);
    vector<float> out(4);
    for (int i = 0; i < 4; i++)
    {
        mat_x[i] = 1;
        mat_y[i] =1;
    }
    mul(mat_x,mat_y,out,2,2,2);
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            printf("%f\t",out[i*2+j]);
        }
        printf("\n");
    }
    return 0;    
}