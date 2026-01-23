#include <torch/extension.h>

torch::Tensor matrix_mul_baseline(torch::Tensor x, torch::Tensor y);
torch::Tensor matrix_mul_v1(torch::Tensor x, torch::Tensor y);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
   m.def("matrix_mul_baseline", torch::wrap_pybind_function(matrix_mul_baseline), "matrix_mul_baseline");
   m.def("matrix_mul_v1", torch::wrap_pybind_function(matrix_mul_v1), "matrix_mul_v1");
}
