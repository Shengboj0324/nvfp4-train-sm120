#include <torch/extension.h>

using OptTensor = c10::optional<torch::Tensor>;

std::vector<torch::Tensor> nvfp4_quantize(torch::Tensor x, bool transpose, bool block_2d, bool stochastic,
                                          int64_t sr_seed, bool use_rht, int64_t rht_seed);
std::vector<torch::Tensor> nvfp4_quantize_dual(torch::Tensor x, bool do_row, bool do_col, bool block_2d,
                                               bool row_sr, int64_t row_seed, bool col_sr, int64_t col_seed,
                                               bool col_rht, int64_t rht_seed, int64_t col_pad,
                                               const OptTensor& row_seed_t, const OptTensor& col_seed_t, const OptTensor& rht_seed_t);
std::vector<torch::Tensor> nvfp4_layernorm_quantize_dual(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta,
                                                         double eps, bool do_row, bool do_col, bool row_sr,
                                                         int64_t row_seed, bool col_sr, int64_t col_seed, bool col_rht,
                                                         int64_t rht_seed, int64_t col_pad, const OptTensor& row_seed_t,
                                                         const OptTensor& col_seed_t, const OptTensor& rht_seed_t);
std::vector<torch::Tensor> nvfp4_gelu_quantize_dual(torch::Tensor h, bool do_row, bool do_col, bool row_sr,
                                                    int64_t row_seed, bool col_sr, int64_t col_seed, bool col_rht,
                                                    int64_t rht_seed, int64_t col_pad, const OptTensor& row_seed_t,
                                                    const OptTensor& col_seed_t, const OptTensor& rht_seed_t);
std::vector<torch::Tensor> nvfp4_gelu_backward_quantize_dual(torch::Tensor dy, torch::Tensor h, bool do_row,
                                                             bool do_col, bool row_sr, int64_t row_seed, bool col_sr,
                                                             int64_t col_seed, bool col_rht, int64_t rht_seed,
                                                             int64_t col_pad, const OptTensor& row_seed_t, const OptTensor& col_seed_t,
                                                             const OptTensor& rht_seed_t);
torch::Tensor nvfp4_gemm(torch::Tensor a, torch::Tensor sfa, torch::Tensor gdec_a, torch::Tensor b, torch::Tensor sfb,
                         torch::Tensor gdec_b, c10::ScalarType out_dtype, c10::optional<torch::Tensor> out,
                         bool accumulate, int64_t impl, int64_t splits);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "NVFP4 kernels for sm_120 (consumer Blackwell)";
  m.def("quantize", &nvfp4_quantize, "BF16/FP32 -> NVFP4 (data, scales, gdec)", py::arg("x"),
        py::arg("transpose") = false, py::arg("block_2d") = false, py::arg("stochastic") = false,
        py::arg("sr_seed") = 0, py::arg("use_rht") = false, py::arg("rht_seed") = 0);
  m.def("quantize_dual", &nvfp4_quantize_dual,
        "Fused rowwise + columnwise (transposed, optional RHT, padded) NVFP4 quantization from one read",
        py::arg("x"), py::arg("do_row") = true, py::arg("do_col") = true, py::arg("block_2d") = false,
        py::arg("row_sr") = false, py::arg("row_seed") = 0, py::arg("col_sr") = false, py::arg("col_seed") = 0,
        py::arg("col_rht") = false, py::arg("rht_seed") = 0, py::arg("col_pad") = 16, py::arg("row_seed_t") = py::none(), py::arg("col_seed_t") = py::none(),
        py::arg("rht_seed_t") = py::none());
  m.def("layernorm_quantize_dual", &nvfp4_layernorm_quantize_dual,
        "LayerNorm fused into quantize_dual: returns mean, rstd and the quantize_dual outputs of LayerNorm(x)",
        py::arg("x"), py::arg("gamma"), py::arg("beta"), py::arg("eps"), py::arg("do_row") = true,
        py::arg("do_col") = true, py::arg("row_sr") = false, py::arg("row_seed") = 0, py::arg("col_sr") = false,
        py::arg("col_seed") = 0, py::arg("col_rht") = false, py::arg("rht_seed") = 0, py::arg("col_pad") = 16, py::arg("row_seed_t") = py::none(), py::arg("col_seed_t") = py::none(),
        py::arg("rht_seed_t") = py::none());
  m.def("gelu_quantize_dual", &nvfp4_gelu_quantize_dual, "tanh-GELU fused into quantize_dual", py::arg("h"),
        py::arg("do_row") = true, py::arg("do_col") = true, py::arg("row_sr") = false, py::arg("row_seed") = 0,
        py::arg("col_sr") = false, py::arg("col_seed") = 0, py::arg("col_rht") = false, py::arg("rht_seed") = 0,
        py::arg("col_pad") = 16, py::arg("row_seed_t") = py::none(), py::arg("col_seed_t") = py::none(),
        py::arg("rht_seed_t") = py::none());
  m.def("gelu_backward_quantize_dual", &nvfp4_gelu_backward_quantize_dual,
        "GELU backward (dy * gelu'(h)) fused into quantize_dual", py::arg("dy"), py::arg("h"),
        py::arg("do_row") = true, py::arg("do_col") = true, py::arg("row_sr") = false, py::arg("row_seed") = 0,
        py::arg("col_sr") = false, py::arg("col_seed") = 0, py::arg("col_rht") = false, py::arg("rht_seed") = 0,
        py::arg("col_pad") = 16, py::arg("row_seed_t") = py::none(), py::arg("col_seed_t") = py::none(),
        py::arg("rht_seed_t") = py::none());
  m.def("gemm", &nvfp4_gemm, "NVFP4 TN GEMM: D = A @ B^T", py::arg("a"), py::arg("sfa"), py::arg("gdec_a"),
        py::arg("b"), py::arg("sfb"), py::arg("gdec_b"), py::arg("out_dtype"), py::arg("out") = py::none(),
        py::arg("accumulate") = false, py::arg("impl") = 0, py::arg("splits") = 0);
}
