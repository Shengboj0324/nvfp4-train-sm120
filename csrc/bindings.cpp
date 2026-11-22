#include <torch/extension.h>

std::vector<torch::Tensor> nvfp4_quantize(torch::Tensor x, bool transpose, bool block_2d, bool stochastic,
                                          int64_t sr_seed, bool use_rht, int64_t rht_seed);
torch::Tensor nvfp4_gemm(torch::Tensor a, torch::Tensor sfa, torch::Tensor gdec_a, torch::Tensor b, torch::Tensor sfb,
                         torch::Tensor gdec_b, c10::ScalarType out_dtype);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "NVFP4 kernels for sm_120 (consumer Blackwell)";
  m.def("quantize", &nvfp4_quantize, "BF16/FP32 -> NVFP4 (data, scales, gdec)", py::arg("x"),
        py::arg("transpose") = false, py::arg("block_2d") = false, py::arg("stochastic") = false,
        py::arg("sr_seed") = 0, py::arg("use_rht") = false, py::arg("rht_seed") = 0);
  m.def("gemm", &nvfp4_gemm, "NVFP4 TN GEMM: D = A @ B^T", py::arg("a"), py::arg("sfa"), py::arg("gdec_a"),
        py::arg("b"), py::arg("sfb"), py::arg("gdec_b"), py::arg("out_dtype"));
}
