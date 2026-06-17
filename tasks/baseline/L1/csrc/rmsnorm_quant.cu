// Fused RMSNorm + per-token-group FP8 quantization CUDA kernel.
//
// Combines rmsnorm and per-token-group FP8 quantization into a single
// kernel launch, eliminating the intermediate BF16 tensor and one kernel
// launch per decoder layer.  This matches the fusion that vLLM's Inductor
// pass (RMSNormQuantFusionPass) achieves.
//
// Two variants:
//   rmsnorm_fp8_quant     — standalone RMSNorm -> FP8 quant
//   fused_add_rmsnorm_fp8_quant — residual-add + RMSNorm -> FP8 quant

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp8.h>
#include <torch/all.h>
#include "utils.h"

constexpr int FP8_GROUP_SIZE = 128;
constexpr float FP8_E4M3_MAX = 448.0f;

template <typename scalar_t>
__global__ void rmsnorm_fp8_quant_kernel(
    __nv_fp8_e4m3* __restrict__ output_fp8,
    float* __restrict__ output_scales,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const float eps,
    const int hidden_size,
    const int num_groups,
    const int scales_stride) {
  const int token = blockIdx.x;
  const scalar_t* x = input + token * hidden_size;

  // Phase 1: compute RMS inverse
  float sum_sq = 0.0f;
  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    float v = static_cast<float>(x[i]);
    sum_sq += v * v;
  }

  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    sum_sq += __shfl_xor_sync(FULL_MASK, sum_sq, offset);

  __shared__ float shared[32];
  int lane = threadIdx.x % WARP_SIZE;
  int wid = threadIdx.x / WARP_SIZE;
  if (lane == 0) shared[wid] = sum_sq;
  __syncthreads();

  sum_sq = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0.0f;
  if (wid == 0) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
      sum_sq += __shfl_xor_sync(FULL_MASK, sum_sq, offset);
  }

  __shared__ float s_rms_inv;
  if (threadIdx.x == 0) {
    s_rms_inv = rsqrtf(sum_sq / hidden_size + eps);
  }
  __syncthreads();

  // Phase 2: RMSNorm + per-group FP8 quantization
  __nv_fp8_e4m3* out_fp8 = output_fp8 + token * hidden_size;

  for (int g = threadIdx.x; g < num_groups; g += blockDim.x) {
    int start = g * FP8_GROUP_SIZE;
    int end_idx = min(start + FP8_GROUP_SIZE, hidden_size);

    // Compute group absmax over normalized values
    float absmax = 0.0f;
    for (int i = start; i < end_idx; i++) {
      float v = static_cast<float>(x[i]) * s_rms_inv *
                static_cast<float>(weight[i]);
      float av = fabsf(v);
      absmax = fmaxf(absmax, av);
    }

    // UE8M0 power-of-two scale
    float scale = exp2f(ceilf(log2f(fmaxf(absmax, 1e-10f) / FP8_E4M3_MAX)));
    float inv_scale = 1.0f / scale;

    // Quantize
    for (int i = start; i < end_idx; i++) {
      float v = static_cast<float>(x[i]) * s_rms_inv *
                static_cast<float>(weight[i]);
      float clamped = fminf(fmaxf(v * inv_scale, -FP8_E4M3_MAX), FP8_E4M3_MAX);
      out_fp8[i] = static_cast<__nv_fp8_e4m3>(clamped);
    }

    // Store scale (column-major layout: scales[group, token])
    output_scales[g * scales_stride + token] = scale;
  }
}

void rmsnorm_fp8_quant(
    torch::Tensor& output_fp8,
    torch::Tensor& output_scales,
    torch::Tensor& input,
    torch::Tensor& weight,
    double eps) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  int num_groups = (hidden_size + FP8_GROUP_SIZE - 1) / FP8_GROUP_SIZE;
  int scales_stride = output_scales.stride(-1);

  dim3 grid(num_tokens);
  dim3 block(std::min(std::max(hidden_size, num_groups), 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_FLOAT_TYPES(input.scalar_type(), "rmsnorm_fp8_quant_kernel", [&] {
    rmsnorm_fp8_quant_kernel<scalar_t><<<grid, block, 0, stream>>>(
        reinterpret_cast<__nv_fp8_e4m3*>(output_fp8.data_ptr()),
        output_scales.data_ptr<float>(),
        input.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        static_cast<float>(eps),
        hidden_size,
        num_groups,
        scales_stride);
  });
}

template <typename scalar_t>
__global__ void fused_add_rmsnorm_fp8_quant_kernel(
    __nv_fp8_e4m3* __restrict__ output_fp8,
    float* __restrict__ output_scales,
    scalar_t* __restrict__ input,
    scalar_t* __restrict__ residual,
    const scalar_t* __restrict__ weight,
    const float eps,
    const int hidden_size,
    const int num_groups,
    const int scales_stride) {
  const int token = blockIdx.x;
  scalar_t* x = input + token * hidden_size;
  scalar_t* r = residual + token * hidden_size;

  // Phase 1: residual += input, compute RMS inverse
  float sum_sq = 0.0f;
  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    float ri = static_cast<float>(r[i]) + static_cast<float>(x[i]);
    r[i] = static_cast<scalar_t>(ri);
    sum_sq += ri * ri;
  }

  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    sum_sq += __shfl_xor_sync(FULL_MASK, sum_sq, offset);

  __shared__ float shared[32];
  int lane = threadIdx.x % WARP_SIZE;
  int wid = threadIdx.x / WARP_SIZE;
  if (lane == 0) shared[wid] = sum_sq;
  __syncthreads();

  sum_sq = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0.0f;
  if (wid == 0) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
      sum_sq += __shfl_xor_sync(FULL_MASK, sum_sq, offset);
  }

  __shared__ float s_rms_inv;
  if (threadIdx.x == 0) {
    s_rms_inv = rsqrtf(sum_sq / hidden_size + eps);
  }
  __syncthreads();

  // Phase 2: RMSNorm(residual) + per-group FP8 quant
  __nv_fp8_e4m3* out_fp8 = output_fp8 + token * hidden_size;

  for (int g = threadIdx.x; g < num_groups; g += blockDim.x) {
    int start = g * FP8_GROUP_SIZE;
    int end_idx = min(start + FP8_GROUP_SIZE, hidden_size);

    float absmax = 0.0f;
    for (int i = start; i < end_idx; i++) {
      float v = static_cast<float>(r[i]) * s_rms_inv *
                static_cast<float>(weight[i]);
      absmax = fmaxf(absmax, fabsf(v));
    }

    float scale = exp2f(ceilf(log2f(fmaxf(absmax, 1e-10f) / FP8_E4M3_MAX)));
    float inv_scale = 1.0f / scale;

    for (int i = start; i < end_idx; i++) {
      float v = static_cast<float>(r[i]) * s_rms_inv *
                static_cast<float>(weight[i]);
      float clamped = fminf(fmaxf(v * inv_scale, -FP8_E4M3_MAX), FP8_E4M3_MAX);
      out_fp8[i] = static_cast<__nv_fp8_e4m3>(clamped);
    }

    output_scales[g * scales_stride + token] = scale;
  }

  // Also write the normalized output back to input (for the next layer's
  // use, matching the non-fused code path where RMSNorm writes to input).
  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    x[i] = static_cast<scalar_t>(
        static_cast<float>(r[i]) * s_rms_inv * static_cast<float>(weight[i]));
  }
}

void fused_add_rmsnorm_fp8_quant(
    torch::Tensor& output_fp8,
    torch::Tensor& output_scales,
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor weight,
    double eps) {
  CHECK_INPUT(input);
  CHECK_INPUT(residual);
  CHECK_INPUT(weight);
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  int num_groups = (hidden_size + FP8_GROUP_SIZE - 1) / FP8_GROUP_SIZE;
  int scales_stride = output_scales.stride(-1);

  dim3 grid(num_tokens);
  dim3 block(std::min(std::max(hidden_size, num_groups), 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_FLOAT_TYPES(input.scalar_type(), "fused_add_rmsnorm_fp8_quant_kernel", [&] {
    fused_add_rmsnorm_fp8_quant_kernel<scalar_t><<<grid, block, 0, stream>>>(
        reinterpret_cast<__nv_fp8_e4m3*>(output_fp8.data_ptr()),
        output_scales.data_ptr<float>(),
        input.data_ptr<scalar_t>(),
        residual.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        static_cast<float>(eps),
        hidden_size,
        num_groups,
        scales_stride);
  });
}
