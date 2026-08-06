// ATen shim replacing vLLM's stable-ABI torch_utils.h so vendored kernels
// compile inside fastkernels' classic torch::Tensor JIT extension.
#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <deque>
#include <mutex>
#include <string>
#include <vector>

// Prefer torch's STD_TORCH_CHECK if already defined (header-only Exception.h).
#ifndef STD_TORCH_CHECK
#define STD_TORCH_CHECK TORCH_CHECK
#endif
#ifndef STD_TORCH_CHECK_NOT_IMPLEMENTED
#define STD_TORCH_CHECK_NOT_IMPLEMENTED(cond, ...) \
  TORCH_CHECK(cond, "NotImplementedError: ", ##__VA_ARGS__)
#endif
#ifndef TORCH_UTILS_CHECK
#define TORCH_UTILS_CHECK STD_TORCH_CHECK
#endif
#ifndef STD_CUDA_CHECK
#define STD_CUDA_CHECK(EXPR)                                          \
  do {                                                                \
    const cudaError_t __err = (EXPR);                                 \
    STD_TORCH_CHECK(__err == cudaSuccess, "CUDA error: ",             \
                    cudaGetErrorString(__err));                       \
  } while (0)
#endif
#ifndef STD_CUDA_KERNEL_LAUNCH_CHECK
#define STD_CUDA_KERNEL_LAUNCH_CHECK() STD_CUDA_CHECK(cudaGetLastError())
#endif

namespace torch {
namespace stable {

using Tensor = at::Tensor;
using Device = at::Device;

inline Tensor contiguous(const Tensor& t) { return t.contiguous(); }

inline Tensor flatten(const Tensor& t, int64_t start_dim = 0,
                      int64_t end_dim = -1) {
  return t.flatten(start_dim, end_dim);
}

inline Tensor empty(at::IntArrayRef size, at::ScalarType dtype,
                    std::optional<at::ScalarType> /*layout*/,
                    at::Device device) {
  return at::empty(size, at::TensorOptions().dtype(dtype).device(device));
}

namespace accelerator {
using DeviceGuard = c10::cuda::CUDAGuard;
}  // namespace accelerator

}  // namespace stable
}  // namespace torch

inline cudaStream_t get_current_cuda_stream(int32_t device_index = -1) {
  if (device_index < 0) {
    return at::cuda::getCurrentCUDAStream().stream();
  }
  return at::cuda::getCurrentCUDAStream(device_index).stream();
}

inline cudaStream_t get_current_cuda_stream(at::Device device) {
  return get_current_cuda_stream(device.index());
}

inline cublasHandle_t get_current_cuda_blas_handle() {
  return at::cuda::getCurrentCUDABlasHandle();
}

// Device properties cache (mirrors vLLM's stable-ABI torch_utils.h).
inline std::deque<std::once_flag> device_flags;
inline std::vector<cudaDeviceProp> device_properties;
inline std::once_flag vectors_init_flag;

inline void do_init_device_vectors() {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  STD_TORCH_CHECK(err == cudaSuccess,
                  "cudaGetDeviceCount failed: ", cudaGetErrorString(err));
  device_flags.resize(device_count);
  device_properties.resize(device_count);
}

inline void initDeviceVectors() {
  std::call_once(vectors_init_flag, do_init_device_vectors);
}

inline void initDeviceProperty(int device_index) {
  cudaDeviceProp device_prop{};
  cudaError_t err = cudaGetDeviceProperties(&device_prop, device_index);
  STD_TORCH_CHECK(err == cudaSuccess,
                  "cudaGetDeviceProperties failed: ", cudaGetErrorString(err));
  device_properties[device_index] = device_prop;
}

inline cudaDeviceProp* get_device_prop() {
  initDeviceVectors();
  int device_index = 0;
  cudaError_t err = cudaGetDevice(&device_index);
  STD_TORCH_CHECK(err == cudaSuccess,
                  "cudaGetDevice failed: ", cudaGetErrorString(err));
  STD_TORCH_CHECK(device_index >= 0 &&
                      static_cast<size_t>(device_index) < device_properties.size(),
                  "CUDA device index out of range");
  std::call_once(device_flags[device_index], initDeviceProperty, device_index);
  return &device_properties[device_index];
}
