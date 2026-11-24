#include <torch/extension.h>

using OptTensor = c10::optional<torch::Tensor>;

std::vector<torch::Tensor> nvfp4_quantize(torch::Tensor x, bool transpose, bool block_2d, bool stochastic,
                                          int64_t sr_seed, bool use_rht, int64_t rht_seed);
std::vector<torch::Tensor> nvfp4_quantize_dual(torch::Tensor x, bool do_row, bool do_col, bool block_2d,
                                               bool row_sr, int64_t row_seed, bool col_sr, int64_t col_seed,
                                               bool col_rht, int64_t rht_seed, int64_t col_pad,
                                               const OptTensor& row_seed_t, const OptTensor& col_seed_t, const OptTensor& rht_seed_t);
std::vector<torch::Tensor> nvfp4_quantize_dual_parts(std::vector<torch::Tensor> parts, bool do_row, bool do_col,
                                                     bool row_sr, int64_t row_seed, bool col_sr, int64_t col_seed,
                                                     bool col_rht, int64_t rht_seed, int64_t col_pad,
                                                     const OptTensor& row_seed_t, const OptTensor& col_seed_t,
                                                     const OptTensor& rht_seed_t, bool col_sums);
std::vector<torch::Tensor> nvfp4_layernorm_quantize_dual(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta,
                                                         double eps, bool do_row, bool do_col, bool row_sr,
                                                         int64_t row_seed, bool col_sr, int64_t col_seed, bool col_rht,
                                                         int64_t rht_seed, int64_t col_pad, const OptTensor& row_seed_t,
                                                         const OptTensor& col_seed_t, const OptTensor& rht_seed_t);
std::vector<torch::Tensor> nvfp4_gelu_quantize_dual(torch::Tensor h, bool do_row, bool do_col, bool row_sr,
                                                    int64_t row_seed, bool col_sr, int64_t col_seed, bool col_rht,
                                                    int64_t rht_seed, int64_t col_pad, const OptTensor& row_seed_t,
                                                    const OptTensor& col_seed_t, const OptTensor& rht_seed_t,
                                                    const OptTensor& bias, bool exact);
std::vector<torch::Tensor> nvfp4_gelu_backward_quantize_dual(torch::Tensor dy, torch::Tensor h, bool do_row,
                                                             bool do_col, bool row_sr, int64_t row_seed, bool col_sr,
                                                             int64_t col_seed, bool col_rht, int64_t rht_seed,
                                                             int64_t col_pad, const OptTensor& row_seed_t, const OptTensor& col_seed_t,
                                                             const OptTensor& rht_seed_t, const OptTensor& bias, bool exact,
                                                             bool col_sums);
std::vector<torch::Tensor> nvfp4_rmsnorm_quantize_dual(torch::Tensor x, torch::Tensor gamma, double eps, bool do_row,
                                                       bool do_col, bool row_sr, int64_t row_seed, bool col_sr,
                                                       int64_t col_seed, bool col_rht, int64_t rht_seed, int64_t col_pad,
                                                       const OptTensor& row_seed_t, const OptTensor& col_seed_t,
                                                       const OptTensor& rht_seed_t);
std::vector<torch::Tensor> nvfp4_swiglu_quantize_dual(torch::Tensor g, torch::Tensor u, bool do_row, bool do_col,
                                                      bool row_sr, int64_t row_seed, bool col_sr, int64_t col_seed,
                                                      bool col_rht, int64_t rht_seed, int64_t col_pad,
                                                      const OptTensor& row_seed_t, const OptTensor& col_seed_t,
                                                      const OptTensor& rht_seed_t, const OptTensor& bias_g,
                                                      const OptTensor& bias_u);
std::vector<torch::Tensor> nvfp4_swiglu_backward_quantize_dual(torch::Tensor dy, torch::Tensor g, torch::Tensor u,
                                                               bool gate, bool do_row, bool do_col, bool row_sr,
                                                               int64_t row_seed, bool col_sr, int64_t col_seed,
                                                               bool col_rht, int64_t rht_seed, int64_t col_pad,
                                                               const OptTensor& row_seed_t, const OptTensor& col_seed_t,
                                                               const OptTensor& rht_seed_t, const OptTensor& bias_g,
                                                               const OptTensor& bias_u, bool col_sums);
std::vector<torch::Tensor> nvfp4_norm_backward(torch::Tensor dy, torch::Tensor x, const OptTensor& mean,
                                               torch::Tensor rstd, torch::Tensor w, const OptTensor& dres, bool need_dx,
                                               bool need_dw, bool need_db);
torch::Tensor nvfp4_gemm(torch::Tensor a, torch::Tensor sfa, torch::Tensor gdec_a, torch::Tensor b, torch::Tensor sfb,
                         torch::Tensor gdec_b, c10::ScalarType out_dtype, c10::optional<torch::Tensor> out,
                         bool accumulate, int64_t impl, int64_t splits, c10::optional<torch::Tensor> bias,
                         c10::optional<torch::Tensor> residual, bool bias_round);

void nvfp4_set_four_six(bool on);
bool nvfp4_four_six();
void nvfp4_set_ms_eden(bool on);
bool nvfp4_ms_eden();
std::vector<torch::Tensor> nvfp4_attn_quant(torch::Tensor x, const OptTensor& mean, const OptTensor& o, bool want_row,
                                            bool want_col, bool want_bf16, bool want_deq);
std::vector<torch::Tensor> nvfp4_attn_fwd(torch::Tensor q, torch::Tensor qs, torch::Tensor k, torch::Tensor ks,
                                          torch::Tensor vt, torch::Tensor vts, int64_t N, bool causal, double scale,
                                          c10::ScalarType dtype, bool p2);
std::vector<torch::Tensor> nvfp4_attn_bwd(torch::Tensor q, torch::Tensor qs, torch::Tensor qt, torch::Tensor qts,
                                          torch::Tensor k, torch::Tensor ks, torch::Tensor kt, torch::Tensor kts,
                                          torch::Tensor k16, torch::Tensor v, torch::Tensor vs, torch::Tensor g,
                                          torch::Tensor gs, torch::Tensor gt, torch::Tensor gts, torch::Tensor lse,
                                          torch::Tensor delta, int64_t N, bool causal, double scale,
                                          c10::ScalarType dtype, const std::vector<int64_t>& dq_stride,
                                          const std::vector<int64_t>& dk_stride);

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
  m.def("quantize_dual_parts", &nvfp4_quantize_dual_parts,
        "quantize_dual of the concatenation (dim 1) of 1-3 tensors, read in place (+ per-tile column sums)",
        py::arg("parts"), py::arg("do_row") = true, py::arg("do_col") = true, py::arg("row_sr") = false,
        py::arg("row_seed") = 0, py::arg("col_sr") = false, py::arg("col_seed") = 0, py::arg("col_rht") = false,
        py::arg("rht_seed") = 0, py::arg("col_pad") = 16, py::arg("row_seed_t") = py::none(),
        py::arg("col_seed_t") = py::none(), py::arg("rht_seed_t") = py::none(), py::arg("col_sums") = false);
  m.def("layernorm_quantize_dual", &nvfp4_layernorm_quantize_dual,
        "LayerNorm fused into quantize_dual: returns mean, rstd and the quantize_dual outputs of LayerNorm(x)",
        py::arg("x"), py::arg("gamma"), py::arg("beta"), py::arg("eps"), py::arg("do_row") = true,
        py::arg("do_col") = true, py::arg("row_sr") = false, py::arg("row_seed") = 0, py::arg("col_sr") = false,
        py::arg("col_seed") = 0, py::arg("col_rht") = false, py::arg("rht_seed") = 0, py::arg("col_pad") = 16, py::arg("row_seed_t") = py::none(), py::arg("col_seed_t") = py::none(),
        py::arg("rht_seed_t") = py::none());
  m.def("gelu_quantize_dual", &nvfp4_gelu_quantize_dual, "GELU(h + bias) (tanh or exact) fused into quantize_dual",
        py::arg("h"), py::arg("do_row") = true, py::arg("do_col") = true, py::arg("row_sr") = false,
        py::arg("row_seed") = 0, py::arg("col_sr") = false, py::arg("col_seed") = 0, py::arg("col_rht") = false,
        py::arg("rht_seed") = 0, py::arg("col_pad") = 16, py::arg("row_seed_t") = py::none(),
        py::arg("col_seed_t") = py::none(), py::arg("rht_seed_t") = py::none(), py::arg("bias") = py::none(),
        py::arg("exact") = false);
  m.def("gelu_backward_quantize_dual", &nvfp4_gelu_backward_quantize_dual,
        "GELU backward (dy * gelu'(h + bias)) fused into quantize_dual (+ per-tile column sums)", py::arg("dy"),
        py::arg("h"), py::arg("do_row") = true, py::arg("do_col") = true, py::arg("row_sr") = false,
        py::arg("row_seed") = 0, py::arg("col_sr") = false, py::arg("col_seed") = 0, py::arg("col_rht") = false,
        py::arg("rht_seed") = 0, py::arg("col_pad") = 16, py::arg("row_seed_t") = py::none(),
        py::arg("col_seed_t") = py::none(), py::arg("rht_seed_t") = py::none(), py::arg("bias") = py::none(),
        py::arg("exact") = false, py::arg("col_sums") = false);
  m.def("rmsnorm_quantize_dual", &nvfp4_rmsnorm_quantize_dual,
        "RMSNorm fused into quantize_dual: returns rstd and the quantize_dual outputs of RMSNorm(x)", py::arg("x"),
        py::arg("gamma"), py::arg("eps"), py::arg("do_row") = true, py::arg("do_col") = true, py::arg("row_sr") = false, py::arg("row_seed") = 0,
        py::arg("col_sr") = false, py::arg("col_seed") = 0, py::arg("col_rht") = false, py::arg("rht_seed") = 0,
        py::arg("col_pad") = 16, py::arg("row_seed_t") = py::none(), py::arg("col_seed_t") = py::none(),
        py::arg("rht_seed_t") = py::none());
  m.def("swiglu_quantize_dual", &nvfp4_swiglu_quantize_dual,
        "SwiGLU (silu(g + bias_g) * (u + bias_u)) fused into quantize_dual", py::arg("g"), py::arg("u"),
        py::arg("do_row") = true, py::arg("do_col") = true, py::arg("row_sr") = false, py::arg("row_seed") = 0,
        py::arg("col_sr") = false, py::arg("col_seed") = 0, py::arg("col_rht") = false, py::arg("rht_seed") = 0,
        py::arg("col_pad") = 16, py::arg("row_seed_t") = py::none(), py::arg("col_seed_t") = py::none(),
        py::arg("rht_seed_t") = py::none(), py::arg("bias_g") = py::none(), py::arg("bias_u") = py::none());
  m.def("swiglu_backward_quantize_dual", &nvfp4_swiglu_backward_quantize_dual,
        "SwiGLU backward (gradient w.r.t. g if gate else u) fused into quantize_dual", py::arg("dy"), py::arg("g"),
        py::arg("u"), py::arg("gate"), py::arg("do_row") = true, py::arg("do_col") = true, py::arg("row_sr") = false, py::arg("row_seed") = 0,
        py::arg("col_sr") = false, py::arg("col_seed") = 0, py::arg("col_rht") = false, py::arg("rht_seed") = 0,
        py::arg("col_pad") = 16, py::arg("row_seed_t") = py::none(), py::arg("col_seed_t") = py::none(),
        py::arg("rht_seed_t") = py::none(), py::arg("bias_g") = py::none(), py::arg("bias_u") = py::none(),
        py::arg("col_sums") = false);
  m.def("norm_backward", &nvfp4_norm_backward,
        "LayerNorm (mean given) / RMSNorm backward from the fused quantizers' statistics, + residual gradient",
        py::arg("dy"), py::arg("x"), py::arg("mean"), py::arg("rstd"), py::arg("w"), py::arg("dres") = py::none(),
        py::arg("need_dx") = true, py::arg("need_dw") = true, py::arg("need_db") = false);
  m.def("gemm", &nvfp4_gemm, "NVFP4 TN GEMM: D = A @ B^T", py::arg("a"), py::arg("sfa"), py::arg("gdec_a"),
        py::arg("b"), py::arg("sfb"), py::arg("gdec_b"), py::arg("out_dtype"), py::arg("out") = py::none(),
        py::arg("accumulate") = false, py::arg("impl") = 0, py::arg("splits") = 0, py::arg("bias") = py::none(),
        py::arg("residual") = py::none(), py::arg("bias_round") = false);
  m.def("set_four_six", &nvfp4_set_four_six, "Enable / disable Four Over Six block scales on round-to-nearest NVFP4 outputs",
        py::arg("on"));
  m.def("four_six", &nvfp4_four_six, "Whether Four Over Six block scales are enabled");
  m.def("set_ms_eden", &nvfp4_set_ms_eden,
        "Enable / disable MS-EDEN (64-point RHT, EDEN-corrected scales instead of element SR) on wgrad operands",
        py::arg("on"));
  m.def("ms_eden", &nvfp4_ms_eden, "Whether MS-EDEN is enabled");
  m.def("attn_quant", &nvfp4_attn_quant,
        "Attention operand quantization: x [B, H, N, D] -> int8 row / MXFP8 col (transposed, permuted) layouts (+ delta)",
        py::arg("x"), py::arg("mean") = py::none(), py::arg("o") = py::none(), py::arg("want_row") = true,
        py::arg("want_col") = true, py::arg("want_bf16") = false, py::arg("want_deq") = false);
  m.def("attn_fwd", &nvfp4_attn_fwd, "8-bit flash attention forward -> (o [B, N, H, D], lse)", py::arg("q"),
        py::arg("qs"), py::arg("k"), py::arg("ks"), py::arg("vt"), py::arg("vts"), py::arg("N"), py::arg("causal"),
        py::arg("scale"), py::arg("dtype"), py::arg("p2") = false);
  m.def("attn_bwd", &nvfp4_attn_bwd, "8-bit flash attention backward -> (dq, dk, dv), [B, H(k), N, D] with the given strides",
        py::arg("q"), py::arg("qs"), py::arg("qt"), py::arg("qts"), py::arg("k"), py::arg("ks"), py::arg("kt"), py::arg("kts"), py::arg("k16"),
        py::arg("v"), py::arg("vs"), py::arg("g"), py::arg("gs"), py::arg("gt"), py::arg("gts"),
        py::arg("lse"), py::arg("delta"), py::arg("N"), py::arg("causal"), py::arg("scale"), py::arg("dtype"),
        py::arg("dq_stride") = std::vector<int64_t>{}, py::arg("dk_stride") = std::vector<int64_t>{});
}
