// fastkernels custom CUDA ops – PyTorch extension binding.
#include <torch/extension.h>

// Forward declarations — legacy / local kernels
void rmsnorm(torch::Tensor& output, torch::Tensor& input, torch::Tensor& weight, double eps);
void fused_add_rmsnorm(torch::Tensor input, torch::Tensor residual, torch::Tensor weight, double eps);
void moe_sum(torch::Tensor& input, torch::Tensor& output);
void moe_align_block_size(torch::Tensor topk_ids, int64_t num_experts, int64_t block_size,
                          torch::Tensor sorted_token_ids, torch::Tensor experts_ids,
                          torch::Tensor num_tokens_post_pad, torch::Tensor cumsum_buffer,
                          bool pad_sorted_token_ids);
void topk_softmax(torch::Tensor& topk_weights, torch::Tensor& topk_indices,
                  torch::Tensor& gating_output, bool renormalize, double moe_softcapping,
                  const c10::optional<torch::Tensor>& correction_bias);
void rmsnorm_fp8_quant(torch::Tensor& output_fp8, torch::Tensor& output_scales,
                       torch::Tensor& input, torch::Tensor& weight, double eps);
void fused_add_rmsnorm_fp8_quant(torch::Tensor& output_fp8, torch::Tensor& output_scales,
                                 torch::Tensor input, torch::Tensor residual,
                                 torch::Tensor weight, double eps);
void build_tree_kernel_efficient(at::Tensor parent_list, at::Tensor selected_index,
                                 at::Tensor verified_seq_len, at::Tensor tree_mask,
                                 at::Tensor positions, at::Tensor retrive_index,
                                 at::Tensor retrive_next_token,
                                 at::Tensor retrive_next_sibling, int64_t topk,
                                 int64_t depth, int64_t draft_token_num,
                                 int64_t tree_mask_mode);
void build_tree_kernel_efficient_with_metadata(
    at::Tensor parent_list, at::Tensor selected_index,
    at::Tensor verified_seq_len, at::Tensor positions,
    at::Tensor retrive_index, at::Tensor retrive_next_token,
    at::Tensor retrive_next_sibling, at::Tensor slot_mapping,
    at::Tensor page_table_expand, at::Tensor cache_seqlens_expand,
    int64_t topk, int64_t depth, int64_t draft_token_num);
void verify_tree_greedy(at::Tensor predicts, at::Tensor accept_index,
                        at::Tensor accept_token_num, at::Tensor candidates,
                        at::Tensor retrive_index, at::Tensor retrive_next_token,
                        at::Tensor retrive_next_sibling, at::Tensor target_predict);
void build_tree_cascade_metadata(at::Tensor tree_mask, at::Tensor slot_mapping,
                                 at::Tensor page_table_expand,
                                 at::Tensor cache_seqlens_expand,
                                 int64_t draft_token_num);

// DeepSeek-V3 router ops (ported verbatim from vLLM csrc/moe).
void dsv3_router_gemm(at::Tensor& output, const at::Tensor& mat_a,
                      const at::Tensor& mat_b);
torch::Tensor router_gemm_bf16_fp32(torch::Tensor const& input,
                                    torch::Tensor const& weight);
std::tuple<torch::Tensor, torch::Tensor> grouped_topk(
    torch::Tensor const& scores, int64_t n_group, int64_t topk_group,
    int64_t topk, bool renormalize, double routed_scaling_factor,
    torch::Tensor const& bias, int64_t scoring_func);

// ---- Vendored vLLM 0.26 kernels (bit-identical with torch.ops._C / _C_cache_ops) ----
void rms_norm(torch::Tensor& out, torch::Tensor& input,
              std::optional<torch::Tensor> weight, double epsilon);
void fused_add_rms_norm(torch::Tensor& input, torch::Tensor& residual,
                        std::optional<torch::Tensor> weight, double epsilon);
void silu_and_mul(torch::Tensor& out, torch::Tensor& input);
void gelu_and_mul(torch::Tensor& out, torch::Tensor& input);
void gelu_tanh_and_mul(torch::Tensor& out, torch::Tensor& input);
void rotary_embedding(torch::Tensor& positions, torch::Tensor& query,
                      std::optional<torch::Tensor> key, int64_t head_size,
                      torch::Tensor& cos_sin_cache, bool is_neox,
                      int64_t rope_dim_offset, bool inverse);
void per_token_group_quant_fp8(const torch::Tensor& input,
                               torch::Tensor& output_q,
                               torch::Tensor& output_s, int64_t group_size,
                               double eps, double fp8_min, double fp8_max,
                               bool scale_ue8m0,
                               bool dummy_is_scale_transposed,
                               bool dummy_is_tma_aligned);
void static_scaled_fp8_quant(
    torch::Tensor& out, torch::Tensor const& input, torch::Tensor const& scale,
    std::optional<at::IntArrayRef> opt_group_shape);
void scaled_fp4_quant_out(torch::Tensor const& input,
                          torch::Tensor const& input_sf,
                          bool is_sf_swizzled_layout, torch::Tensor& output,
                          torch::Tensor& output_sf);
void top_k_per_row_prefill(const torch::Tensor& logits,
                           const torch::Tensor& rowStarts,
                           const torch::Tensor& rowEnds, torch::Tensor& indices,
                           int64_t numRows, int64_t stride0, int64_t stride1,
                           int64_t topK);
void top_k_per_row_decode(const torch::Tensor& logits, int64_t next_n,
                          const torch::Tensor& seqLens, torch::Tensor& indices,
                          int64_t numRows, int64_t stride0, int64_t stride1,
                          int64_t topK);
void persistent_topk(const torch::Tensor& logits, const torch::Tensor& lengths,
                     torch::Tensor& output, torch::Tensor& workspace, int64_t k,
                     int64_t max_seq_len);
void cooperative_topk(const torch::Tensor& logits, const torch::Tensor& lengths,
                      torch::Tensor& output, torch::Tensor& workspace, int64_t k,
                      int64_t max_seq_len);
void concat_and_cache_mla(torch::Tensor& kv_c, torch::Tensor& k_pe,
                          torch::Tensor& kv_cache, torch::Tensor& slot_mapping,
                          const std::string& kv_cache_dtype,
                          torch::Tensor& scale);
void gather_and_maybe_dequant_cache(
    torch::Tensor const& src_cache, torch::Tensor const& dst,
    torch::Tensor const& block_table, torch::Tensor const& cu_seq_lens,
    torch::Tensor const& token_to_seq, int64_t num_tokens,
    const std::string& kv_cache_dtype, torch::Tensor const& scale,
    std::optional<torch::Tensor> seq_starts);
void cp_gather_and_upconvert_fp8_kv_cache(
    torch::Tensor const& src_cache, torch::Tensor const& dst,
    torch::Tensor const& block_table, torch::Tensor const& workspace_starts,
    int64_t batch_size, std::optional<torch::Tensor> seq_starts);
void indexer_k_quant_and_cache(torch::Tensor& k, torch::Tensor& kv_cache,
                               torch::Tensor& slot_mapping,
                               int64_t quant_block_size,
                               const std::string& scale_fmt);
void cp_gather_indexer_k_quant_cache(torch::Tensor const& kv_cache,
                                     torch::Tensor& dst_k,
                                     torch::Tensor& dst_scale,
                                     torch::Tensor const& block_table,
                                     torch::Tensor const& cu_seq_lens);
void merge_attn_states(torch::Tensor& output,
                       std::optional<torch::Tensor> output_lse,
                       const torch::Tensor& prefix_output,
                       const torch::Tensor& prefix_lse,
                       const torch::Tensor& suffix_output,
                       const torch::Tensor& suffix_lse,
                       const std::optional<int64_t> prefill_tokens_with_context,
                       const std::optional<torch::Tensor>& output_scale);
void selective_scan_fwd(
    const torch::Tensor& u, const torch::Tensor& delta, const torch::Tensor& A,
    const torch::Tensor& B, const torch::Tensor& C,
    const std::optional<torch::Tensor>& D_,
    const std::optional<torch::Tensor>& z_,
    const std::optional<torch::Tensor>& delta_bias_, bool delta_softplus,
    const std::optional<torch::Tensor>& query_start_loc,
    const std::optional<torch::Tensor>& cache_indices,
    const std::optional<torch::Tensor>& has_initial_state,
    const torch::Tensor& ssm_states, int64_t null_block_id, int64_t block_size,
    const std::optional<torch::Tensor>& block_idx_first_scheduled_token,
    const std::optional<torch::Tensor>& block_idx_last_scheduled_token,
    const std::optional<torch::Tensor>& initial_state_idx,
    const std::optional<torch::Tensor>& cu_chunk_seqlen,
    const std::optional<torch::Tensor>& last_chunk_indices);

// Default-arg wrappers for pybind (match vLLM's schema defaults).
static void rotary_embedding_py(
    torch::Tensor& positions, torch::Tensor& query,
    std::optional<torch::Tensor> key, int64_t head_size,
    torch::Tensor& cos_sin_cache, bool is_neox,
    int64_t rope_dim_offset = 0, bool inverse = false) {
  rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox,
                   rope_dim_offset, inverse);
}

static void per_token_group_fp8_quant_py(
    const torch::Tensor& input, torch::Tensor& output_q,
    torch::Tensor& output_s, int64_t group_size, double eps, double fp8_min,
    double fp8_max, bool scale_ue8m0, bool is_scale_transposed = false,
    bool is_tma_aligned = false) {
  per_token_group_quant_fp8(input, output_q, output_s, group_size, eps, fp8_min,
                            fp8_max, scale_ue8m0, is_scale_transposed,
                            is_tma_aligned);
}

static void static_scaled_fp8_quant_py(
    torch::Tensor& out, torch::Tensor const& input, torch::Tensor const& scale,
    std::optional<at::IntArrayRef> opt_group_shape = std::nullopt) {
  static_scaled_fp8_quant(out, input, scale, opt_group_shape);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Legacy local names (still used by fastkernels_norm torch.library wrappers).
  m.def("rmsnorm", &rmsnorm, "RMSNorm (CUDA, local)");
  m.def("fused_add_rmsnorm", &fused_add_rmsnorm, "Fused add + RMSNorm (CUDA, local)");
  m.def("moe_sum", &moe_sum, "MoE sum reduction (CUDA)");
  m.def("moe_align_block_size", &moe_align_block_size, "MoE align block size (CUDA)");
  m.def("topk_softmax", &topk_softmax, "Top-K softmax for MoE (CUDA)");
  m.def("rmsnorm_fp8_quant", &rmsnorm_fp8_quant, "Fused RMSNorm + FP8 quant (CUDA)");
  m.def("fused_add_rmsnorm_fp8_quant", &fused_add_rmsnorm_fp8_quant, "Fused add + RMSNorm + FP8 quant (CUDA)");
  m.def("build_tree_kernel_efficient", &build_tree_kernel_efficient,
        "EAGLE build tree kernel efficient (CUDA)");
  m.def("build_tree_kernel_efficient_with_metadata",
        &build_tree_kernel_efficient_with_metadata,
        "EAGLE build tree and FA3 metadata kernel efficient (CUDA)");
  m.def("verify_tree_greedy", &verify_tree_greedy, "EAGLE verify tree greedy (CUDA)");
  m.def("build_tree_cascade_metadata", &build_tree_cascade_metadata,
        "EAGLE build FA3 cascade metadata (CUDA)");
  m.def("dsv3_router_gemm", &dsv3_router_gemm,
        "DeepSeek-V3 router GEMM (SM90+, BF16->{FP32,BF16}) (CUDA)");
  m.def("router_gemm_bf16_fp32", &router_gemm_bf16_fp32,
        "cuBLAS BF16xBF16->FP32 router GEMM fallback (CUDA)");
  m.def("grouped_topk", &grouped_topk,
        "Fused noaux_tc grouped top-k for MoE routing (CUDA)");

  // vLLM-parity op names (replace torch.ops._C / _C_cache_ops).
  m.def("rms_norm", &rms_norm, "vLLM RMSNorm (CUDA)");
  m.def("fused_add_rms_norm", &fused_add_rms_norm, "vLLM fused add+RMSNorm (CUDA)");
  m.def("silu_and_mul", &silu_and_mul, "vLLM silu_and_mul (CUDA)");
  m.def("gelu_and_mul", &gelu_and_mul, "vLLM gelu_and_mul (CUDA)");
  m.def("gelu_tanh_and_mul", &gelu_tanh_and_mul, "vLLM gelu_tanh_and_mul (CUDA)");
  m.def("rotary_embedding", &rotary_embedding_py, "vLLM rotary_embedding (CUDA)",
        py::arg("positions"), py::arg("query"), py::arg("key"),
        py::arg("head_size"), py::arg("cos_sin_cache"), py::arg("is_neox"),
        py::arg("rope_dim_offset") = 0, py::arg("inverse") = false);
  m.def("per_token_group_fp8_quant", &per_token_group_fp8_quant_py,
        "vLLM per_token_group_fp8_quant (CUDA)",
        py::arg("input"), py::arg("output_q"), py::arg("output_s"),
        py::arg("group_size"), py::arg("eps"), py::arg("fp8_min"),
        py::arg("fp8_max"), py::arg("scale_ue8m0"),
        py::arg("is_scale_transposed") = false,
        py::arg("is_tma_aligned") = false);
  m.def("static_scaled_fp8_quant", &static_scaled_fp8_quant_py,
        "vLLM static_scaled_fp8_quant (CUDA)",
        py::arg("out"), py::arg("input"), py::arg("scale"),
        py::arg("group_shape") = py::none());
  m.def("scaled_fp4_quant_out", &scaled_fp4_quant_out,
        "vLLM scaled_fp4_quant.out (CUDA)");
  m.def("top_k_per_row_prefill", &top_k_per_row_prefill,
        "vLLM top_k_per_row_prefill (CUDA)");
  m.def("top_k_per_row_decode", &top_k_per_row_decode,
        "vLLM top_k_per_row_decode (CUDA)");
  m.def("persistent_topk", &persistent_topk, "vLLM persistent_topk (CUDA)");
  m.def("cooperative_topk", &cooperative_topk, "vLLM cooperative_topk (CUDA)");
  m.def("concat_and_cache_mla", &concat_and_cache_mla,
        "vLLM concat_and_cache_mla (CUDA)");
  m.def("gather_and_maybe_dequant_cache", &gather_and_maybe_dequant_cache,
        "vLLM gather_and_maybe_dequant_cache (CUDA)");
  m.def("cp_gather_and_upconvert_fp8_kv_cache",
        &cp_gather_and_upconvert_fp8_kv_cache,
        "vLLM cp_gather_and_upconvert_fp8_kv_cache (CUDA)");
  m.def("indexer_k_quant_and_cache", &indexer_k_quant_and_cache,
        "vLLM indexer_k_quant_and_cache (CUDA)");
  m.def("cp_gather_indexer_k_quant_cache", &cp_gather_indexer_k_quant_cache,
        "vLLM cp_gather_indexer_k_quant_cache (CUDA)");
  m.def("merge_attn_states", &merge_attn_states,
        "vLLM merge_attn_states (CUDA)", py::arg("output"),
        py::arg("output_lse"), py::arg("prefix_output"), py::arg("prefix_lse"),
        py::arg("suffix_output"), py::arg("suffix_lse"),
        py::arg("prefill_tokens_with_context"), py::arg("output_scale"));
  m.def("selective_scan_fwd", &selective_scan_fwd,
        "vLLM Mamba selective_scan_fwd (CUDA)");
}
