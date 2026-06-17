// kb_nano custom CUDA ops – PyTorch extension binding.
#include <torch/extension.h>

// Forward declarations
void rmsnorm(torch::Tensor& output, torch::Tensor& input, torch::Tensor& weight, double eps);
void fused_add_rmsnorm(torch::Tensor input, torch::Tensor residual, torch::Tensor weight, double eps);
void silu_and_mul(at::Tensor& out, at::Tensor& input);
void rotary_embedding(torch::Tensor& positions, torch::Tensor& query,
                      std::optional<torch::Tensor> key, int64_t head_size,
                      torch::Tensor& cos_sin_cache, bool is_neox);
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
//
// ``dsv3_router_gemm`` is the SM90+ specialised BF16xBF16->{FP32,BF16}
// gate matmul (num_tokens<=16, num_experts in {256,384}, hidden=7168).
//
// ``router_gemm_bf16_fp32`` is the cuBLAS BF16xBF16->FP32 fallback used
// for batches > 16.
//
// ``grouped_topk`` is the fully-fused noaux_tc grouped top-k kernel
// (sigmoid + grouped top-k + e_score_correction_bias + renormalize +
// scaling) returning (topk_values, topk_indices) as a 2-tensor tuple.
void dsv3_router_gemm(at::Tensor& output, const at::Tensor& mat_a,
                      const at::Tensor& mat_b);
torch::Tensor router_gemm_bf16_fp32(torch::Tensor const& input,
                                    torch::Tensor const& weight);
std::tuple<torch::Tensor, torch::Tensor> grouped_topk(
    torch::Tensor const& scores, int64_t n_group, int64_t topk_group,
    int64_t topk, bool renormalize, double routed_scaling_factor,
    torch::Tensor const& bias, int64_t scoring_func);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rmsnorm", &rmsnorm, "RMSNorm (CUDA)");
  m.def("fused_add_rmsnorm", &fused_add_rmsnorm, "Fused add + RMSNorm (CUDA)");
  m.def("silu_and_mul", &silu_and_mul, "SiLU and Mul activation (CUDA)");
  m.def("rotary_embedding", &rotary_embedding, "Rotary position embedding (CUDA)");
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
}
