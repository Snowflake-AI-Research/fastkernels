// Reuse GeluAndMul's actual CUDA launch, vector loads/stores, and product.
// Only its activation callback changes from GELU to identity.
#include <torch/extension.h>

// Include the unmodified parent implementation without exporting its module.
#pragma push_macro("PYBIND11_MODULE")
#undef PYBIND11_MODULE
#define PYBIND11_MODULE(name, module) \
  static void unused_parent_bindings(pybind11::module_& module)
#include "../../tasks/baseline/L1/gelu_and_mul.cu"
#pragma pop_macro("PYBIND11_MODULE")

template <typename T>
__device__ __forceinline__ T identity_gate(const T& value, const float) {
  return value;
}

void product_gate(torch::Tensor& out, torch::Tensor& input) {
  LAUNCH_ACTIVATION_GATE_KERNEL(identity_gate, identity_gate,
                               true, false, 0.0f, 1.0f, 0.0f);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("product_gate", &product_gate);
}
