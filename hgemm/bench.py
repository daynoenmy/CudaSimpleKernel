import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
lib = load(
    name="mul_lib",
    sources=["main.cpp","hgemm.cu"],
    extra_cuda_cflags=['-O2'],
    extra_cflags=["-std=c++17"]
)
def hgemm_naive(x,y):
    return x@y

M = 256
N=128
K =32
x = torch.rand(M,K,device="cuda",dtype=torch.float16)
y = torch.rand(K,N,device="cuda",dtype=torch.float16)
z = hgemm_naive(x,y)
my_out = lib.matrix_mul_baseline(x,y)
# baseline的误差会随着K的增大而增大
# 128 时 dif >1e-2
print('softmax values sanity check:', torch.allclose(z, my_out, rtol=0, atol=1e-02))
diff = (z - my_out).abs()
idx = diff.argmax()
print("最大误差:", diff.max().item())