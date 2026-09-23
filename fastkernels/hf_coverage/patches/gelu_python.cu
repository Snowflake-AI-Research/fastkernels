// Retain the parent's actual activation launch and vector/scalar memory paths.
#include <torch/extension.h>

#pragma push_macro("PYBIND11_MODULE")
#undef PYBIND11_MODULE
#define PYBIND11_MODULE(name, module) \
  static void unused_parent_bindings(pybind11::module_& module)
#include "../../tasks/baseline/L1/gelu_and_mul.cu"
#pragma pop_macro("PYBIND11_MODULE")

template <typename T>
__device__ __forceinline__ T rounded_gelu(const T& input) {
  const float value = static_cast<float>(input);
  // Each T cast represents the result of one HF tensor operation. Explicit
  // FP32 round-to-nearest intrinsics also keep separate operations unfused.
  const T half_input = static_cast<T>(__fmul_rn(value, 0.5f));
  const T scaled = static_cast<T>(__fmul_rn(value, static_cast<float>(M_SQRT1_2)));
  const T error_function = static_cast<T>(erff(static_cast<float>(scaled)));
  const T shifted = static_cast<T>(__fadd_rn(1.0f, static_cast<float>(error_function)));
  return static_cast<T>(__fmul_rn(static_cast<float>(half_input), static_cast<float>(shifted)));
}

void gelu_python(torch::Tensor& out, torch::Tensor& input) {
  LAUNCH_ACTIVATION_KERNEL(rounded_gelu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("gelu_python", &gelu_python);
}
