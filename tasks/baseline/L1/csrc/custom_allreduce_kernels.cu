#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include <array>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#define CUDACHECK(cmd)                                              \
  do {                                                              \
    cudaError_t e = cmd;                                            \
    if (e != cudaSuccess) {                                         \
      throw std::runtime_error(                                     \
          std::string("CUDA error: ") + cudaGetErrorString(e) +     \
          " at " + __FILE__ + ":" + std::to_string(__LINE__));      \
    }                                                               \
  } while (0)

constexpr int kMaxBlocks = 36;
const int defaultBlockLimit = 36;

using FlagType = uint32_t;
using fptr_t = int64_t;
static_assert(sizeof(void*) == sizeof(fptr_t));

struct Signal {
  alignas(128) FlagType start[kMaxBlocks][8];
  alignas(128) FlagType end[kMaxBlocks][8];
  alignas(128) FlagType _flag[kMaxBlocks];
};

struct __align__(16) RankData {
  const void* ptrs[8];
};

struct __align__(16) RankSignals {
  Signal* signals[8];
};

template <typename T, int sz>
struct __align__(alignof(T) * sz) array_t {
  T data[sz];
  using type = T;
  static constexpr int size = sz;
};

template <typename T>
struct packed_t {
  using P = array_t<T, 16 / sizeof(T)>;
  using A = array_t<float, 16 / sizeof(T)>;
};

#define DINLINE __device__ __forceinline__

DINLINE float upcast_s(half val) { return __half2float(val); }
DINLINE float upcast_s(nv_bfloat16 val) { return __bfloat162float(val); }

template <typename T>
DINLINE T downcast_s(float val);
template <>
DINLINE half downcast_s(float val) { return __float2half(val); }
template <>
DINLINE nv_bfloat16 downcast_s(float val) { return __float2bfloat16(val); }

DINLINE half& assign_add(half& a, half b) { a = __hadd(a, b); return a; }
DINLINE nv_bfloat16& assign_add(nv_bfloat16& a, nv_bfloat16 b) {
  a = __hadd(a, b); return a;
}
DINLINE float& assign_add(float& a, float b) { return a += b; }

template <typename T, int N>
DINLINE array_t<T, N>& packed_assign_add(array_t<T, N>& a, array_t<T, N> b) {
#pragma unroll
  for (int i = 0; i < N; i++) assign_add(a.data[i], b.data[i]);
  return a;
}

template <typename T, int N>
DINLINE array_t<float, N> upcast(array_t<T, N> val) {
  if constexpr (std::is_same<T, float>::value) {
    return val;
  } else {
    array_t<float, N> out;
#pragma unroll
    for (int i = 0; i < N; i++) out.data[i] = upcast_s(val.data[i]);
    return out;
  }
}

template <typename O>
DINLINE O downcast(array_t<float, O::size> val) {
  if constexpr (std::is_same<typename O::type, float>::value) {
    return val;
  } else {
    O out;
#pragma unroll
    for (int i = 0; i < O::size; i++)
      out.data[i] = downcast_s<typename O::type>(val.data[i]);
    return out;
  }
}

// ---------------------------------------------------------------------------
// Flag operations for synchronization (NVIDIA GPUs, SM >= 7.0)
// ---------------------------------------------------------------------------
static DINLINE void st_flag_release(FlagType* flag_addr, FlagType flag) {
  asm volatile("st.release.sys.global.u32 [%1], %0;" ::"r"(flag),
               "l"(flag_addr));
}

static DINLINE FlagType ld_flag_acquire(FlagType* flag_addr) {
  FlagType flag;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
               : "=r"(flag)
               : "l"(flag_addr));
  return flag;
}

static DINLINE void st_flag_volatile(FlagType* flag_addr, FlagType flag) {
  asm volatile("st.volatile.global.u32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
}

static DINLINE FlagType ld_flag_volatile(FlagType* flag_addr) {
  FlagType flag;
  asm volatile("ld.volatile.global.u32 %0, [%1];"
               : "=r"(flag)
               : "l"(flag_addr));
  return flag;
}

// ---------------------------------------------------------------------------
// Barriers
// ---------------------------------------------------------------------------
template <int ngpus>
DINLINE void barrier_at_start(const RankSignals& sg, Signal* self_sg,
                              int rank) {
  uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
  if (threadIdx.x < ngpus) {
    auto peer_counter_ptr = &sg.signals[threadIdx.x]->start[blockIdx.x][rank];
    auto self_counter_ptr = &self_sg->start[blockIdx.x][threadIdx.x];
    st_flag_volatile(peer_counter_ptr, flag);
    while (ld_flag_volatile(self_counter_ptr) != flag);
  }
  __syncthreads();
  if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}

template <int ngpus, bool final_sync = false>
DINLINE void barrier_at_end(const RankSignals& sg, Signal* self_sg, int rank) {
  __syncthreads();
  uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
  if (threadIdx.x < ngpus) {
    auto peer_counter_ptr = &sg.signals[threadIdx.x]->end[blockIdx.x][rank];
    auto self_counter_ptr = &self_sg->end[blockIdx.x][threadIdx.x];
    if constexpr (!final_sync) {
      st_flag_release(peer_counter_ptr, flag);
      while (ld_flag_acquire(self_counter_ptr) != flag);
    } else {
      st_flag_volatile(peer_counter_ptr, flag);
      while (ld_flag_volatile(self_counter_ptr) != flag);
    }
  }
  if constexpr (!final_sync) __syncthreads();
  if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}

// ---------------------------------------------------------------------------
// Reduction kernels
// ---------------------------------------------------------------------------
template <typename P, int ngpus, typename A>
DINLINE P packed_reduce(const P* ptrs[], int idx) {
  A tmp = upcast(ptrs[0][idx]);
#pragma unroll
  for (int i = 1; i < ngpus; i++) {
    packed_assign_add(tmp, upcast(ptrs[i][idx]));
  }
  return downcast<P>(tmp);
}

template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1)
    cross_device_reduce_1stage(RankData* _dp, RankSignals sg, Signal* self_sg,
                               T* __restrict__ result, int rank, int size) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  auto dp = *_dp;
  barrier_at_start<ngpus>(sg, self_sg, rank);
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += gridDim.x * blockDim.x) {
    ((P*)result)[idx] =
        packed_reduce<P, ngpus, A>((const P**)&dp.ptrs[0], idx);
  }
  barrier_at_end<ngpus, true>(sg, self_sg, rank);
}

template <typename P>
DINLINE P* get_tmp_buf(Signal* sg) {
  return (P*)(((Signal*)sg) + 1);
}

template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1)
    cross_device_reduce_2stage(RankData* _dp, RankSignals sg, Signal* self_sg,
                               T* __restrict__ result, int rank, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  int part = size / ngpus;
  int start = rank * part;
  int end = rank == ngpus - 1 ? size : start + part;
  int largest_part = part + size % ngpus;
  const P* ptrs[ngpus];
  P* tmps[ngpus];
#pragma unroll
  for (int i = 0; i < ngpus; i++) {
    int target = (rank + i) % ngpus;
    ptrs[i] = (const P*)_dp->ptrs[target];
    tmps[i] = get_tmp_buf<P>(sg.signals[target]);
  }
  auto tmp_out = tmps[0];
  barrier_at_start<ngpus>(sg, self_sg, rank);
  for (int idx = start + tid; idx < end; idx += stride) {
    tmp_out[idx - start] = packed_reduce<P, ngpus, A>(ptrs, idx);
  }
  barrier_at_end<ngpus>(sg, self_sg, rank);
  for (int idx = tid; idx < largest_part; idx += stride) {
#pragma unroll
    for (int i = 0; i < ngpus; i++) {
      int gather_from_rank = ((rank + i) % ngpus);
      if (gather_from_rank == ngpus - 1 || idx < part) {
        int dst_idx = gather_from_rank * part + idx;
        ((P*)result)[dst_idx] = tmps[i][idx];
      }
    }
  }
}

// ---------------------------------------------------------------------------
// IPC handle type
// ---------------------------------------------------------------------------
using IPC_KEY = std::array<uint8_t, sizeof(cudaIpcMemHandle_t)>;
static_assert(sizeof(IPC_KEY) == sizeof(cudaIpcMemHandle_t));
static_assert(alignof(IPC_KEY) == alignof(cudaIpcMemHandle_t));

CUpointer_attribute rangeStartAddrAttr = CU_POINTER_ATTRIBUTE_RANGE_START_ADDR;

// ---------------------------------------------------------------------------
// C++ CustomAllreduce class
// ---------------------------------------------------------------------------
class CustomAllreduce {
 public:
  int rank_;
  int world_size_;
  bool fully_connected_;

  RankSignals sg_;
  std::unordered_map<void*, RankData*> buffers_;
  Signal* self_sg_;

  RankData *d_rank_data_base_, *d_rank_data_end_;
  std::vector<void*> graph_unreg_buffers_;
  std::map<IPC_KEY, char*> ipc_handles_;

  CustomAllreduce(Signal** signals, void* rank_data, size_t rank_data_sz,
                  int rank, int world_size, bool fully_connected = true)
      : rank_(rank),
        world_size_(world_size),
        fully_connected_(fully_connected),
        self_sg_(signals[rank]),
        d_rank_data_base_(reinterpret_cast<RankData*>(rank_data)),
        d_rank_data_end_(d_rank_data_base_ + rank_data_sz / sizeof(RankData)) {
    for (int i = 0; i < world_size_; i++) {
      sg_.signals[i] = signals[i];
    }
  }

  char* open_ipc_handle(const void* ipc_handle) {
    auto [it, new_handle] =
        ipc_handles_.insert({*((IPC_KEY*)ipc_handle), nullptr});
    if (new_handle) {
      char* ipc_ptr;
      CUDACHECK(cudaIpcOpenMemHandle((void**)&ipc_ptr,
                                     *((const cudaIpcMemHandle_t*)ipc_handle),
                                     cudaIpcMemLazyEnablePeerAccess));
      it->second = ipc_ptr;
    }
    return it->second;
  }

  std::pair<std::string, std::vector<int64_t>> get_graph_buffer_ipc_meta() {
    auto num_buffers = graph_unreg_buffers_.size();
    auto handle_sz = sizeof(cudaIpcMemHandle_t);
    std::string handles(handle_sz * num_buffers, static_cast<char>(0));
    std::vector<int64_t> offsets(num_buffers);
    for (size_t i = 0; i < num_buffers; i++) {
      auto ptr = graph_unreg_buffers_[i];
      void* base_ptr;
      if (cuPointerGetAttribute(&base_ptr, rangeStartAddrAttr,
                                (CUdeviceptr)ptr) != CUDA_SUCCESS)
        throw std::runtime_error("failed to get pointer attr");
      CUDACHECK(cudaIpcGetMemHandle(
          (cudaIpcMemHandle_t*)&handles[i * handle_sz], base_ptr));
      offsets[i] = ((char*)ptr) - ((char*)base_ptr);
    }
    return std::make_pair(handles, offsets);
  }

  void check_rank_data_capacity(size_t num = 1) {
    if (d_rank_data_base_ + num > d_rank_data_end_)
      throw std::runtime_error(
          "Rank data buffer overflow by " +
          std::to_string(d_rank_data_base_ + num - d_rank_data_end_));
  }

  void register_buffer(void** ptrs) {
    check_rank_data_capacity();
    RankData data;
    for (int i = 0; i < world_size_; i++) {
      data.ptrs[i] = ptrs[i];
    }
    auto d_data = d_rank_data_base_++;
    CUDACHECK(
        cudaMemcpy(d_data, &data, sizeof(RankData), cudaMemcpyHostToDevice));
    buffers_[ptrs[rank_]] = d_data;
  }

  void register_graph_buffers(
      const std::vector<std::string>& handles,
      const std::vector<std::vector<int64_t>>& offsets) {
    auto num_buffers = graph_unreg_buffers_.size();
    check_rank_data_capacity(num_buffers);
    std::vector<RankData> rank_data(num_buffers);
    for (size_t i = 0; i < num_buffers; i++) {
      auto self_ptr = graph_unreg_buffers_[i];
      auto& rd = rank_data[i];
      for (int j = 0; j < world_size_; j++) {
        if (j != rank_) {
          char* handle =
              open_ipc_handle(&handles[j][i * sizeof(cudaIpcMemHandle_t)]);
          handle += offsets[j][i];
          rd.ptrs[j] = handle;
        } else {
          rd.ptrs[j] = self_ptr;
        }
      }
    }
    CUDACHECK(cudaMemcpy(d_rank_data_base_, rank_data.data(),
                         sizeof(RankData) * num_buffers,
                         cudaMemcpyHostToDevice));
    d_rank_data_base_ += num_buffers;
    graph_unreg_buffers_.clear();
  }

  template <typename T>
  void allreduce(cudaStream_t stream, T* input, T* output, int size,
                 int threads = 512, int block_limit = defaultBlockLimit) {
    auto d = packed_t<T>::P::size;
    if (size % d != 0)
      throw std::runtime_error(
          "custom allreduce requires input length to be multiple of " +
          std::to_string(d));
    if (block_limit > kMaxBlocks)
      throw std::runtime_error("max supported block limit is " +
                               std::to_string(kMaxBlocks));

    RankData* ptrs;
    cudaStreamCaptureStatus status;
    CUDACHECK(cudaStreamIsCapturing(stream, &status));
    if (status == cudaStreamCaptureStatusActive) {
      ptrs = d_rank_data_base_ + graph_unreg_buffers_.size();
      graph_unreg_buffers_.push_back(input);
    } else {
      auto it = buffers_.find(input);
      if (it == buffers_.end())
        throw std::runtime_error(
            "buffer address " +
            std::to_string(reinterpret_cast<uint64_t>(input)) +
            " is not registered!");
      ptrs = it->second;
    }

    size /= d;
    int blocks = std::min(block_limit, (size + threads - 1) / threads);

#define KL(ngpus, name)                                                       \
  name<T, ngpus><<<blocks, threads, 0, stream>>>(ptrs, sg_, self_sg_, output, \
                                                 rank_, size);
#define REDUCE_CASE(ngpus)                              \
  case ngpus: {                                         \
    if (world_size_ == 2) {                             \
      KL(ngpus, cross_device_reduce_1stage);            \
    } else if (fully_connected_) {                      \
      auto bytes = size * sizeof(typename packed_t<T>::P); \
      if ((world_size_ <= 4 && bytes < 512 * 1024) ||   \
          (world_size_ <= 8 && bytes < 256 * 1024)) {   \
        KL(ngpus, cross_device_reduce_1stage);          \
      } else {                                          \
        KL(ngpus, cross_device_reduce_2stage);          \
      }                                                 \
    }                                                   \
    break;                                              \
  }

    switch (world_size_) {
      REDUCE_CASE(2)
      REDUCE_CASE(4)
      REDUCE_CASE(6)
      REDUCE_CASE(8)
      default:
        throw std::runtime_error(
            "custom allreduce only supports world_size in {2,4,6,8}, got " +
            std::to_string(world_size_));
    }
#undef REDUCE_CASE
#undef KL
  }

  ~CustomAllreduce() {
    for (auto [_, ptr] : ipc_handles_) {
      cudaIpcCloseMemHandle(ptr);
    }
  }
};

// ---------------------------------------------------------------------------
// Python-facing functions (torch extension bindings)
// ---------------------------------------------------------------------------

fptr_t init_custom_ar(const std::vector<fptr_t>& fake_ipc_ptrs,
                      torch::Tensor& rank_data, int64_t rank,
                      bool fully_connected) {
  int world_size = fake_ipc_ptrs.size();
  if (world_size > 8)
    throw std::invalid_argument("world size > 8 is not supported");
  if (world_size % 2 != 0)
    throw std::invalid_argument("odd num gpus is not supported");
  if (rank < 0 || rank >= world_size)
    throw std::invalid_argument("invalid rank");

  Signal* ipc_ptrs[8];
  for (int i = 0; i < world_size; i++) {
    ipc_ptrs[i] = reinterpret_cast<Signal*>(fake_ipc_ptrs[i]);
  }
  return (fptr_t) new CustomAllreduce(ipc_ptrs, rank_data.data_ptr(),
                                      rank_data.numel(), rank, world_size,
                                      fully_connected);
}

void all_reduce(fptr_t _fa, torch::Tensor& inp, torch::Tensor& out,
                fptr_t _reg_buffer, int64_t reg_buffer_sz_bytes) {
  auto fa = reinterpret_cast<CustomAllreduce*>(_fa);
  auto stream = c10::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK_EQ(inp.scalar_type(), out.scalar_type());
  TORCH_CHECK_EQ(inp.numel(), out.numel());
  auto input_size = inp.numel() * inp.element_size();
  auto reg_buffer = reinterpret_cast<void*>(_reg_buffer);
  if (reg_buffer) {
    TORCH_CHECK_LE(input_size, reg_buffer_sz_bytes);
    CUDACHECK(cudaMemcpyAsync(reg_buffer, inp.data_ptr(), input_size,
                              cudaMemcpyDeviceToDevice, stream));
  } else {
    reg_buffer = inp.data_ptr();
  }
  switch (out.scalar_type()) {
    case at::ScalarType::Float:
      fa->allreduce<float>(stream, reinterpret_cast<float*>(reg_buffer),
                           reinterpret_cast<float*>(out.data_ptr()),
                           out.numel());
      break;
    case at::ScalarType::Half:
      fa->allreduce<half>(stream, reinterpret_cast<half*>(reg_buffer),
                          reinterpret_cast<half*>(out.data_ptr()), out.numel());
      break;
    case at::ScalarType::BFloat16:
      fa->allreduce<nv_bfloat16>(
          stream, reinterpret_cast<nv_bfloat16*>(reg_buffer),
          reinterpret_cast<nv_bfloat16*>(out.data_ptr()), out.numel());
      break;
    default:
      throw std::runtime_error(
          "custom allreduce only supports float32, float16 and bfloat16");
  }
}

void dispose(fptr_t _fa) {
  delete reinterpret_cast<CustomAllreduce*>(_fa);
}

int64_t meta_size() { return sizeof(Signal); }

void register_buffer(fptr_t _fa, const std::vector<fptr_t>& fake_ipc_ptrs) {
  auto fa = reinterpret_cast<CustomAllreduce*>(_fa);
  TORCH_CHECK(static_cast<int>(fake_ipc_ptrs.size()) == fa->world_size_);
  void* ipc_ptrs[8];
  for (size_t i = 0; i < fake_ipc_ptrs.size(); i++) {
    ipc_ptrs[i] = reinterpret_cast<void*>(fake_ipc_ptrs[i]);
  }
  fa->register_buffer(ipc_ptrs);
}

std::tuple<std::vector<int64_t>, std::vector<int64_t>>
get_graph_buffer_ipc_meta(fptr_t _fa) {
  auto fa = reinterpret_cast<CustomAllreduce*>(_fa);
  auto [handle, offsets] = fa->get_graph_buffer_ipc_meta();
  std::vector<int64_t> bytes(handle.begin(), handle.end());
  return std::make_tuple(bytes, offsets);
}

void register_graph_buffers(fptr_t _fa,
                            const std::vector<std::vector<int64_t>>& handles,
                            const std::vector<std::vector<int64_t>>& offsets) {
  auto fa = reinterpret_cast<CustomAllreduce*>(_fa);
  std::vector<std::string> bytes;
  bytes.reserve(handles.size());
  for (size_t i = 0; i < handles.size(); i++) {
    bytes.emplace_back(handles[i].begin(), handles[i].end());
  }
  fa->register_graph_buffers(bytes, offsets);
}

std::tuple<fptr_t, torch::Tensor> allocate_shared_buffer_and_handle(
    int64_t size) {
  auto device_index = c10::cuda::current_device();
  at::DeviceGuard device_guard(at::Device(at::DeviceType::CUDA, device_index));
  void* buffer;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));

  CUDACHECK(cudaMalloc((void**)&buffer, size));
  CUDACHECK(cudaMemsetAsync(buffer, 0, size, stream));
  CUDACHECK(cudaStreamSynchronize(stream));
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));

  auto options =
      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
  auto handle =
      torch::empty({static_cast<int64_t>(sizeof(cudaIpcMemHandle_t))}, options);
  CUDACHECK(
      cudaIpcGetMemHandle((cudaIpcMemHandle_t*)handle.data_ptr(), buffer));

  return std::make_tuple(reinterpret_cast<fptr_t>(buffer), handle);
}

fptr_t open_mem_handle(torch::Tensor& mem_handle) {
  void* ipc_ptr;
  CUDACHECK(cudaIpcOpenMemHandle(
      (void**)&ipc_ptr, *((const cudaIpcMemHandle_t*)mem_handle.data_ptr()),
      cudaIpcMemLazyEnablePeerAccess));
  return reinterpret_cast<fptr_t>(ipc_ptr);
}

void free_shared_buffer(fptr_t buffer) {
  CUDACHECK(cudaFree(reinterpret_cast<void*>(buffer)));
}

// ---------------------------------------------------------------------------
// PyBind11 module definition
// ---------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init_custom_ar", &init_custom_ar);
  m.def("all_reduce", &all_reduce);
  m.def("dispose", &dispose);
  m.def("meta_size", &meta_size);
  m.def("register_buffer", &register_buffer);
  m.def("get_graph_buffer_ipc_meta", &get_graph_buffer_ipc_meta);
  m.def("register_graph_buffers", &register_graph_buffers);
  m.def("allocate_shared_buffer_and_handle",
        &allocate_shared_buffer_and_handle);
  m.def("open_mem_handle", &open_mem_handle);
  m.def("free_shared_buffer", &free_shared_buffer);
}
