#include<torch/extension.h>

torch::Tensor matrix_mul_baseline(torch::Tensor x,torch::Tensor y);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
   m.def("matrix_mul_baseline",torch::wrap_pybind_function(matrix_mul_baseline),"matrix_mul_baseline");
}
