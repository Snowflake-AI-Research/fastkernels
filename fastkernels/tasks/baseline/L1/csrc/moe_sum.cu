#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include "utils.h"

template <typename scalar_t, int TOPK>
__global__ void moe_sum_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., topk, d]
    const int d) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    scalar_t x = 0.0;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      x += SGLANG_LDG(&input[token_idx * TOPK * d + k * d + idx]);
    }
    out[token_idx * d + idx] = x;
  }
}

// Float-accumulating variant. The ``scalar_t``-accumulating kernel above is
// only safe for the very small TOPK it is instantiated with; for wider TOPK the
// fallback used to be ``at::sum_out``, which accumulates in
// ``at::acc_type<scalar_t>`` (float for bf16/half). This keeps that float
// accumulation width, so it is no less accurate than the fallback, while
// avoiding the generic reduction. Gemma4 (top_k_experts=8) spent 53.9 ms of a
// 256x512 prefill in ``at::native::reduce_kernel`` before this existed.
//
// It is *not* bit-identical to ``at::sum_out``: ATen vectorizes and may reduce
// in a different order, so the low bits differ. On Gemma4 that is enough to
// reshuffle greedy decoding (two runs differing only in this reduction agreed
// on 158/1000 sequences), which is expected for a 26B MoE -- fastkernels and
// vLLM only agree on ~18% of output tokens at baseline. Measured effect on
// alignment against vLLM was inside one standard error; see the audit doc.
template <typename scalar_t, int TOPK>
__global__ void moe_sum_facc_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., topk, d]
    const int d) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    float x = 0.0f;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      x += static_cast<float>(
          SGLANG_LDG(&input[token_idx * TOPK * d + k * d + idx]));
    }
    out[token_idx * d + idx] = static_cast<scalar_t>(x);
  }
}

void moe_sum(
    torch::Tensor& input,   // [num_tokens, topk, hidden_size]
    torch::Tensor& output)  // [num_tokens, hidden_size]
{
  const int hidden_size = input.size(-1);
  const auto num_tokens = output.numel() / hidden_size;
  const int topk = input.size(1);

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(output));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  switch (topk) {
    case 2:
      DISPATCH_FLOAT_TYPES(input.scalar_type(), "moe_sum_kernel", [&] {
        moe_sum_kernel<scalar_t, 2>
            <<<grid, block, 0, stream>>>(output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), hidden_size);
      });
      break;

    case 3:
      DISPATCH_FLOAT_TYPES(input.scalar_type(), "moe_sum_kernel", [&] {
        moe_sum_kernel<scalar_t, 3>
            <<<grid, block, 0, stream>>>(output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), hidden_size);
      });
      break;

    case 4:
      DISPATCH_FLOAT_TYPES(input.scalar_type(), "moe_sum_kernel", [&] {
        moe_sum_kernel<scalar_t, 4>
            <<<grid, block, 0, stream>>>(output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), hidden_size);
      });
      break;

    case 8:
      DISPATCH_FLOAT_TYPES(input.scalar_type(), "moe_sum_facc_kernel", [&] {
        moe_sum_facc_kernel<scalar_t, 8>
            <<<grid, block, 0, stream>>>(output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), hidden_size);
      });
      break;

    default:
      at::sum_out(output, input, 1);
      break;
  }
}
