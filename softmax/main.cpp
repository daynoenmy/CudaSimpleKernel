#include<torch/extension.h>
torch::Tensor safe_softmax_forward(torch::Tensor x);
torch::Tensor online_softmax_cpu_forward(torch::Tensor x);
torch::Tensor online_softmax_f32_forward(torch::Tensor x);
torch::Tensor online_softmax_f32x4_forward(torch::Tensor x);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
    m.def("safe_softmax_forward",torch::wrap_pybind_function(safe_softmax_forward),"safe_softmax_forward");
    m.def("online_softmax_cpu_forward",torch::wrap_pybind_function(online_softmax_cpu_forward),"online_softmax_cpu");
    m.def("online_softmax_f32_forward",torch::wrap_pybind_function(online_softmax_f32_forward),"online_softmax_f32_forward");
    m.def("online_softmax_f32x4_forward",torch::wrap_pybind_function(online_softmax_f32x4_forward),"online_softmax_f32x4_forward");
}
