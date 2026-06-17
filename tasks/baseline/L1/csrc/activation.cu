// Standalone SiLU-and-Mul activation kernel for kb_nano.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include "utils.h"

template <typename T>
__device__ __forceinline__ float to_float(T v) { return static_cast<float>(v); }
template <> __device__ __forceinline__ float to_float<nv_half>(nv_half v) { return __half2float(v); }
template <> __device__ __forceinline__ float to_float<nv_bfloat16>(nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T>
__device__ __forceinline__ T from_float(float v) { return static_cast<T>(v); }
template <> __device__ __forceinline__ nv_half from_float<nv_half>(float v) { return __float2half(v); }
template <> __device__ __forceinline__ nv_bfloat16 from_float<nv_bfloat16>(float v) { return __float2bfloat16(v); }

template <typename T>
__global__ void silu_and_mul_kernel(
    T* __restrict__ out,
    const T* __restrict__ input,
    const int d) {
  const int64_t token_idx = blockIdx.x;
  const T* row = input + token_idx * 2 * d;
  T* out_row = out + token_idx * d;

  for (int i = threadIdx.x; i < d; i += blockDim.x) {
    float x = to_float(row[i]);
    float y = to_float(row[d + i]);
    float silu_x = x / (1.0f + expf(-x));
    out_row[i] = from_float<T>(silu_x * y);
  }
}

void silu_and_mul(at::Tensor& out, at::Tensor& input) {
  int d = input.size(-1) / 2;
  int64_t num_tokens = input.numel() / input.size(-1);
  dim3 grid(num_tokens);
  dim3 block(std::min(d, 1024));

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), c_type, [&] {
    silu_and_mul_kernel<c_type><<<grid, block, 0, stream>>>(
        static_cast<c_type*>(out.data_ptr()),
        static_cast<c_type*>(input.data_ptr()),
        d);
    return true;
  });
}
