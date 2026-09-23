// Adapt from https://github.com/vllm-project/vllm/blob/v0.7.3/csrc/moe/topk_softmax_kernels.cu
// which is originally adapted from
// https://github.com/NVIDIA/TensorRT-LLM/blob/v0.7.1/cpp/tensorrt_llm/kernels/mixtureOfExperts/moe_kernels.cu
/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/


// Adapted from L1/topk_softmax.cu:339-398, preserving its CUB ArgMax.
// Both sentinels become -infinity for raw scores; retain the parent k loop.
// Rows must contain at least k finite candidates. No renormalization.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cub/cub.cuh>
#include <math_constants.h>

template <int TPB>
__launch_bounds__(TPB) __global__ void moeTopK(
    float* inputs_after_softmax,
    const bool* finished,
    float* output,
    int* indices,
    const int num_experts,
    const int k,
    const int start_expert,
    const int end_expert,
    const bool renormalize) {
  using cub_kvp = cub::KeyValuePair<int, float>;
  using BlockReduce = cub::BlockReduce<cub_kvp, TPB>;
  __shared__ typename BlockReduce::TempStorage tmpStorage;

  cub_kvp thread_kvp;
  cub::ArgMax arg_max;

  const int block_row = blockIdx.x;

  const bool row_is_active = finished ? !finished[block_row] : true;
  const int thread_read_offset = blockIdx.x * num_experts;
  float row_sum_for_renormalize = 0;
  for (int k_idx = 0; k_idx < k; ++k_idx) {
    thread_kvp.key = 0;
    thread_kvp.value = -CUDART_INF_F;  // Raw proposal scores may be negative.

    cub_kvp inp_kvp;
    for (int expert = threadIdx.x; expert < num_experts; expert += TPB) {
      const int idx = thread_read_offset + expert;
      inp_kvp.key = expert;
      inp_kvp.value = inputs_after_softmax[idx];
      thread_kvp = arg_max(inp_kvp, thread_kvp);
    }

    const cub_kvp result_kvp = BlockReduce(tmpStorage).Reduce(thread_kvp, arg_max);
    if (threadIdx.x == 0) {
      // Ignore experts the node isn't responsible for with expert parallelism
      const int expert = result_kvp.key;
      const bool node_uses_expert = expert >= start_expert && expert < end_expert;
      const bool should_process_row = row_is_active && node_uses_expert;

      const int idx = k * block_row + k_idx;
      output[idx] = result_kvp.value;
      indices[idx] = should_process_row ? (expert - start_expert) : num_experts;
      assert(indices[idx] >= 0);
      row_sum_for_renormalize += result_kvp.value;
      // The inputs_after_softmax is modified in-place to avoid unnecessary loops for finding the top k-1 value.
      // Raw scores require a sentinel below every finite candidate.
      inputs_after_softmax[thread_read_offset + expert] = -CUDART_INF_F;
    }
    __syncthreads();
  }

  if (renormalize && threadIdx.x == 0) {
    float row_sum_for_renormalize_inv = 1.f / row_sum_for_renormalize;
    for (int k_idx = 0; k_idx < k; ++k_idx) {
      const int idx = k * block_row + k_idx;
      output[idx] = output[idx] * row_sum_for_renormalize_inv;
    }
  }
}

std::vector<at::Tensor> raw_topk(at::Tensor scores, int64_t k) {
  TORCH_CHECK(scores.is_cuda() && scores.scalar_type() == at::kFloat &&
              scores.is_contiguous() && scores.dim() == 2 && k > 0 && k <= scores.size(1),
              "raw_topk requires contiguous CUDA float32 score rows and 0 < k <= columns");
  const c10::cuda::CUDAGuard guard(scores.device());
  auto indices = at::empty({scores.size(0), k}, scores.options().dtype(at::kInt));
  auto values = at::empty({scores.size(0), k}, scores.options());
  moeTopK<256><<<scores.size(0), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      scores.data_ptr<float>(), nullptr, values.data_ptr<float>(),
      indices.data_ptr<int>(), scores.size(1), k, 0, scores.size(1), false);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {values, indices};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("raw_topk", &raw_topk);
}
