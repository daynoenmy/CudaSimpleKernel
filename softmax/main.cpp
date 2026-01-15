#include<torch/extension.h>
torch::Tensor softmax_forward(torch::Tensor x);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
    m.def("softmax_forward",torch::wrap_pybind_function(softmax_forward),"softmax_forward");
}