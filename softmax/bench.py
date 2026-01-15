import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
lib = load(
    name="softmax_lib",
    sources=["main.cpp","safe_softmax.cu","online_softmax.cu"],
    extra_cuda_cflags=['-O2'],
    extra_cflags=["-std=c++17"]
)
def softmax(x):
    y = F.softmax(x,dim=-1)
    return y; 
batch_size = 512
seq_len = 256
input = torch.randn(batch_size,seq_len).cuda()
py_out = softmax(input)

my_out = lib.online_softmax_f32x4_forward(input)

print('softmax values sanity check:', torch.allclose(py_out, my_out, rtol=0, atol=1e-05))
