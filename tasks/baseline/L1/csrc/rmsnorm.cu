// Standalone RMSNorm and fused-add-RMSNorm CUDA kernels for kb_nano.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/cub.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <numeric>
#include <type_traits>
#include <torch/all.h>
#include "utils.h"

namespace {

template <typename scalar_t, size_t vec_size>
struct __align__(vec_size * sizeof(scalar_t)) vec_n_t {
  scalar_t val[vec_size];
};

template <typename torch_type>
struct TypeConvert {
  static constexpr bool exists = false;
};

template <>
struct TypeConvert<float> {
  static constexpr bool exists = true;
  using device_type = float;
  using packed_type = float2;
  __device__ static __forceinline__ float convert(device_type x) { return x; }
};

template <>
struct TypeConvert<c10::Half> {
  static constexpr bool exists = true;
  using device_type = __half;
  using packed_type = __half2;
  __device__ static __forceinline__ float convert(device_type x) {
    return __half2float(x);
  }
  __device__ static __forceinline__ float2 convert(packed_type x) {
    return __half22float2(x);
  }
  __device__ static __forceinline__ device_type convert(float x) {
    return __float2half_rn(x);
  }
  __device__ static __forceinline__ packed_type convert(float2 x) {
    return __float22half2_rn(x);
  }
};

template <>
struct TypeConvert<c10::BFloat16> {
  static constexpr bool exists = true;
  using device_type = __nv_bfloat16;
  using packed_type = __nv_bfloat162;
  __device__ static __forceinline__ float convert(device_type x) {
    return __bfloat162float(x);
  }
  __device__ static __forceinline__ float2 convert(packed_type x) {
    return __bfloat1622float2(x);
  }
  __device__ static __forceinline__ device_type convert(float x) {
    return __float2bfloat16(x);
  }
  __device__ static __forceinline__ packed_type convert(float2 x) {
    return __float22bfloat162_rn(x);
  }
};

template <typename scalar_t, int width>
struct alignas(16) F16Vec {
  static_assert(width > 0 && (width & (width - 1)) == 0);
  using Converter = TypeConvert<scalar_t>;
  using T1 = typename Converter::device_type;
  using T2 = typename Converter::packed_type;
  T1 data[width];

  __device__ F16Vec& operator+=(const F16Vec& other) {
#pragma unroll
    for (int i = 0; i < width; i += 2) {
      if constexpr (std::is_same_v<T2, float2>) {
        data[i] += other.data[i];
        data[i + 1] += other.data[i + 1];
      } else {
        T2 temp{data[i], data[i + 1]};
        temp += T2{other.data[i], other.data[i + 1]};
        data[i] = temp.x;
        data[i + 1] = temp.y;
      }
    }
    return *this;
  }

  __device__ F16Vec& operator*=(const F16Vec& other) {
#pragma unroll
    for (int i = 0; i < width; i += 2) {
      if constexpr (std::is_same_v<T2, float2>) {
        data[i] *= other.data[i];
        data[i + 1] *= other.data[i + 1];
      } else {
        T2 temp{data[i], data[i + 1]};
        temp *= T2{other.data[i], other.data[i + 1]};
        data[i] = temp.x;
        data[i + 1] = temp.y;
      }
    }
    return *this;
  }

  __device__ F16Vec& operator*=(const float scale) {
#pragma unroll
    for (int i = 0; i < width; i += 2) {
      float2 temp_f = Converter::convert(T2{data[i], data[i + 1]});
      temp_f.x *= scale;
      temp_f.y *= scale;
      T2 temp = Converter::convert(temp_f);
      data[i] = temp.x;
      data[i + 1] = temp.y;
    }
    return *this;
  }

  __device__ float sum_squares() const {
    float result = 0.0f;
#pragma unroll
    for (int i = 0; i < width; i += 2) {
      float2 z = Converter::convert(T2{data[i], data[i + 1]});
      result += z.x * z.x + z.y * z.y;
    }
    return result;
  }
};

template <int VEC_SIZE, typename scalar_t, typename VecOp, typename ScalarOp>
__device__ inline void vectorize_read_with_alignment(
    const scalar_t* input,
    int len,
    int tid,
    int stride,
    VecOp&& vec_op,
    ScalarOp&& scalar_op) {
  constexpr int WIDTH = VEC_SIZE * sizeof(scalar_t);
  uintptr_t addr = reinterpret_cast<uintptr_t>(input);
  bool can_vec = ((addr & (WIDTH - 1)) == 0) && ((len & (VEC_SIZE - 1)) == 0);
  if (can_vec) {
    int num_vec = len / VEC_SIZE;
    auto* v_in = reinterpret_cast<const vec_n_t<scalar_t, VEC_SIZE>*>(input);
    for (int i = tid; i < num_vec; i += stride) {
      vec_op(v_in[i]);
    }
    return;
  }
  int misalignment_offset = addr & (WIDTH - 1);
  int alignment_bytes = WIDTH - misalignment_offset;
  int prefix_elems = (alignment_bytes & (WIDTH - 1)) / sizeof(scalar_t);
  prefix_elems = min(prefix_elems, len);
  for (int i = tid; i < prefix_elems; i += stride) {
    scalar_op(input[i]);
  }
  input += prefix_elems;
  len -= prefix_elems;
  int num_vec = len / VEC_SIZE;
  auto* v_in = reinterpret_cast<const vec_n_t<scalar_t, VEC_SIZE>*>(input);
  for (int i = tid; i < num_vec; i += stride) {
    vec_op(v_in[i]);
  }
  int tail_start = num_vec * VEC_SIZE;
  for (int i = tid + tail_start; i < len; i += stride) {
    scalar_op(input[i]);
  }
}

}  // namespace

template <typename scalar_t>
__global__ void rmsnorm_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const float eps,
    const int hidden_size) {
  const int token = blockIdx.x;
  const scalar_t* x = input + token * hidden_size;
  scalar_t* o = out + token * hidden_size;

  float sum_sq = 0.0f;
  auto vec_op = [&sum_sq](const vec_n_t<scalar_t, 8>& vec) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float v = static_cast<float>(vec.val[i]);
      sum_sq += v * v;
    }
  };
  auto scalar_op = [&sum_sq](const scalar_t& val) {
    float v = static_cast<float>(val);
    sum_sq += v * v;
  };
  vectorize_read_with_alignment<8>(x, hidden_size, threadIdx.x, blockDim.x,
                                   vec_op, scalar_op);

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduce_store;
  sum_sq = BlockReduce(reduce_store).Sum(sum_sq);

  __shared__ float s_rms_inv;
  if (threadIdx.x == 0) {
    s_rms_inv = rsqrtf(sum_sq / hidden_size + eps);
  }
  __syncthreads();

  auto* v_in = reinterpret_cast<const vec_n_t<scalar_t, 8>*>(x);
  auto* v_w = reinterpret_cast<const vec_n_t<scalar_t, 8>*>(weight);
  auto* v_out = reinterpret_cast<vec_n_t<scalar_t, 8>*>(o);
  for (int i = threadIdx.x; i < hidden_size / 8; i += blockDim.x) {
    vec_n_t<scalar_t, 8> dst;
    vec_n_t<scalar_t, 8> src1 = v_in[i];
    vec_n_t<scalar_t, 8> src2 = v_w[i];
#pragma unroll
    for (int j = 0; j < 8; j++) {
      float v = static_cast<float>(src1.val[j]);
      dst.val[j] = static_cast<scalar_t>(v * s_rms_inv) * src2.val[j];
    }
    v_out[i] = dst;
  }
}

void rmsnorm(
    torch::Tensor& output,
    torch::Tensor& input,
    torch::Tensor& weight,
    double eps) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  dim3 grid(num_tokens);
  const int max_block_size = (num_tokens < 256) ? 1024 : 256;
  dim3 block(std::min(hidden_size, max_block_size));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_FLOAT_TYPES(input.scalar_type(), "rmsnorm_kernel", [&] {
    rmsnorm_kernel<scalar_t><<<grid, block, 0, stream>>>(
        output.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        static_cast<float>(eps),
        hidden_size);
  });
}

template <typename scalar_t>
__global__ void fused_add_rmsnorm_kernel(
    scalar_t* __restrict__ input,
    scalar_t* __restrict__ residual,
    const scalar_t* __restrict__ weight,
    const float eps,
    const int hidden_size) {
  const int token = blockIdx.x;
  scalar_t* x = input + token * hidden_size;
  scalar_t* r = residual + token * hidden_size;

  // Step 1: residual += input; then compute rms on residual
  float sum_sq = 0.0f;
  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    float ri = static_cast<float>(r[i]) + static_cast<float>(x[i]);
    r[i] = static_cast<scalar_t>(ri);
    sum_sq += ri * ri;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduce_store;
  sum_sq = BlockReduce(reduce_store).Sum(sum_sq);

  __shared__ float s_rms_inv;
  if (threadIdx.x == 0) {
    s_rms_inv = rsqrtf(sum_sq / hidden_size + eps);
  }
  __syncthreads();

  // Step 2: input = rmsnorm(residual) * weight
  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    float ri = static_cast<float>(r[i]);
    x[i] = static_cast<scalar_t>(ri * s_rms_inv) * weight[i];
  }
}

template <typename scalar_t, int width>
__global__ std::enable_if_t<TypeConvert<scalar_t>::exists>
fused_add_rmsnorm_vec_kernel(
    scalar_t* __restrict__ input,
    scalar_t* __restrict__ residual,
    const scalar_t* __restrict__ weight,
    const float eps,
    const int hidden_size,
    const int64_t input_stride) {
  const int vec_hidden_size = hidden_size / width;
  const int64_t vec_input_stride = input_stride / width;
  float sum_sq = 0.0f;

  auto* __restrict__ input_v = reinterpret_cast<F16Vec<scalar_t, width>*>(input);
  auto* __restrict__ residual_v = reinterpret_cast<F16Vec<scalar_t, width>*>(residual);
  auto* __restrict__ weight_v = reinterpret_cast<const F16Vec<scalar_t, width>*>(weight);

  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    int id = blockIdx.x * vec_hidden_size + idx;
    int64_t strided_id = blockIdx.x * vec_input_stride + idx;
    F16Vec<scalar_t, width> temp = input_v[strided_id];
    temp += residual_v[id];
    sum_sq += temp.sum_squares();
    residual_v[id] = temp;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduce_store;
  sum_sq = BlockReduce(reduce_store).Sum(sum_sq);

  __shared__ float s_rms_inv;
  if (threadIdx.x == 0) {
    s_rms_inv = rsqrtf(sum_sq / hidden_size + eps);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    int id = blockIdx.x * vec_hidden_size + idx;
    int64_t strided_id = blockIdx.x * vec_input_stride + idx;
    F16Vec<scalar_t, width> temp = residual_v[id];
    temp *= s_rms_inv;
    temp *= weight_v[idx];
    input_v[strided_id] = temp;
  }
}

void fused_add_rmsnorm(
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor weight,
    double eps) {
  CHECK_INPUT(input);
  CHECK_INPUT(residual);
  CHECK_INPUT(weight);
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  dim3 grid(num_tokens);
  const int max_block_size = (num_tokens < 256) ? 1024 : 256;
  dim3 block(std::min(hidden_size, max_block_size));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  constexpr int vector_width = 8;
  constexpr int req_alignment_bytes = vector_width * 2;
  auto inp_ptr = reinterpret_cast<uintptr_t>(input.data_ptr());
  auto res_ptr = reinterpret_cast<uintptr_t>(residual.data_ptr());
  auto wt_ptr = reinterpret_cast<uintptr_t>(weight.data_ptr());
  bool ptrs_are_aligned = inp_ptr % req_alignment_bytes == 0 &&
                          res_ptr % req_alignment_bytes == 0 &&
                          wt_ptr % req_alignment_bytes == 0;
  bool offsets_are_multiple_of_vector_width =
      hidden_size % vector_width == 0 && input.stride(-2) % vector_width == 0;
  if (ptrs_are_aligned && offsets_are_multiple_of_vector_width &&
      (input.scalar_type() == at::ScalarType::Half ||
       input.scalar_type() == at::ScalarType::BFloat16)) {
    AT_DISPATCH_SWITCH(
        input.scalar_type(), "fused_add_rmsnorm_vec_kernel",
        AT_DISPATCH_CASE(at::ScalarType::Half, [&] {
          fused_add_rmsnorm_vec_kernel<scalar_t, vector_width><<<grid, block, 0, stream>>>(
              input.data_ptr<scalar_t>(),
              residual.data_ptr<scalar_t>(),
              weight.data_ptr<scalar_t>(),
              static_cast<float>(eps),
              hidden_size,
              input.stride(-2));
        })
        AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] {
          fused_add_rmsnorm_vec_kernel<scalar_t, vector_width><<<grid, block, 0, stream>>>(
              input.data_ptr<scalar_t>(),
              residual.data_ptr<scalar_t>(),
              weight.data_ptr<scalar_t>(),
              static_cast<float>(eps),
              hidden_size,
              input.stride(-2));
        }));
    return;
  }

  DISPATCH_FLOAT_TYPES(input.scalar_type(), "fused_add_rmsnorm_kernel", [&] {
    fused_add_rmsnorm_kernel<scalar_t><<<grid, block, 0, stream>>>(
        input.data_ptr<scalar_t>(),
        residual.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        static_cast<float>(eps),
        hidden_size);
  });
}
