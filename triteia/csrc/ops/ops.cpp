#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <torch/python.h>
#include <torch/library.h>

#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>

#include "sgmv.h"
#include "bgmv_config.h"


#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)

//====== utils ======

inline void check_shape(const torch::Tensor& a, const torch::Tensor& b,
                        const char* a_name, const char* b_name) {
  TORCH_CHECK(a.dim() == b.dim(), a_name, ".dim() != ", b_name, ".dim(). ",
              a.dim(), " vs ", b.dim());
  for (int i = 0; i < a.dim(); ++i) {
    TORCH_CHECK(a.size(i) == b.size(i), a_name, ".size(", i, ") != ", b_name,
                ".size(", i, ")");
  }
}

inline constexpr uint32_t pack_u16(uint16_t a, uint16_t b) {
  return (uint32_t(a) << 16) | uint32_t(b);
}

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

#define CHECK_DIM(d, x) \
  TORCH_CHECK(x.dim() == d, #x " must be a " #d "D tensor")

#define CHECK_SHAPE(a, b) check_shape(a, b, #a, #b)

#define CHECK_EQ(a, b) \
  TORCH_CHECK((a) == (b), "CHECK_EQ(" #a ", " #b ") failed. ", a, " vs ", b)

#define CHECK_GE(a, b) \
  TORCH_CHECK((a) >= (b), "CHECK_GE(" #a ", " #b ") failed. ", a, " vs ", b)

//====== dispatch pytorch dtype ======

#define _DISPATCH_SWITCH(cond, ...) \
  [&]() -> bool {                   \
    switch (cond) {                 \
      __VA_ARGS__                   \
      default:                      \
        return false;               \
    }                               \
  }()

#define _DISPATCH_DTYPE_CASE(enum_type, c_type_, ...) \
  case enum_type: {                                   \
    using c_type = c_type_;                           \
    return __VA_ARGS__();                             \
  }

#define _DISPATCH_DTYPE_CASES(...)                                 \
  _DISPATCH_DTYPE_CASE(at::ScalarType::Half, nv_half, __VA_ARGS__) \
  _DISPATCH_DTYPE_CASE(at::ScalarType::BFloat16, nv_bfloat16, __VA_ARGS__)

#define DISPATCH_TORCH_DTYPE(scalar_type, ...) \
  _DISPATCH_SWITCH(scalar_type, _DISPATCH_DTYPE_CASES(__VA_ARGS__))

namespace marlin {
int marlin_cuda_2_4(const void *A, const void *B, const void *meta, void *C,
                    void *s, int prob_m, int prob_n, int prob_k,
                    void *workspace, int groupsize = -1, int dev = 0,
                    cudaStream_t stream = 0, int thread_k = -1,
                    int thread_m = -1, int sms = -1, int max_par = 16);
const int ERR_PROB_SHAPE = 1;
const int ERR_KERN_SHAPE = 2;
void mul_2_4(const torch::Tensor &A, const torch::Tensor &B,
             const torch::Tensor &meta, torch::Tensor &C,
             const torch::Tensor &s, torch::Tensor &workspace,
             int thread_k = -1, int thread_m = -1, int sms = -1,
             int max_par = 8) {
  int prob_n = A.size(0);
  int prob_m = C.size(1);
  int prob_k = A.size(1);
  int groupsize = (s.size(0) == 1) ? -1 : prob_k / 2 / s.size(0);
  // printf("groupsize is:%d\n", groupsize);
  if (groupsize != -1 && groupsize * s.size(0) != (prob_k / 2))
    AT_ERROR("k=", prob_k, " not compatible with ", s.size(0), " groups.");
  if (workspace.numel() < prob_m / 128 * max_par)
    AT_ERROR("workspace must be of size at least ", prob_m / 128 * max_par,
             ".");
  int dev = A.get_device();
  int err = marlin_cuda_2_4(
      A.data_ptr(), B.data_ptr(), meta.data_ptr(), C.data_ptr(), s.data_ptr(),
      prob_m, prob_n, prob_k, workspace.data_ptr(), groupsize, dev,
      at::cuda::getCurrentCUDAStream(dev), thread_k, thread_m, sms, max_par);
  if (err == ERR_PROB_SHAPE) {
    AT_ERROR("Problem (m=", prob_m, ", n=", prob_n, ", k=", prob_k, ")",
             " not compatible with thread_k=", thread_k,
             ", thread_m=", thread_m, ".");
  } else if (err == ERR_KERN_SHAPE) {
    AT_ERROR("No kernel implementation for thread_k=", thread_k,
             ", thread_m=", thread_m, ", groupsize=", groupsize, ".");
  }
}
}  // namespace marlin

namespace punica {
  void dispatch_sgmv_cutlass(torch::Tensor y, torch::Tensor x,
                           torch::Tensor w_ptr, torch::Tensor s,
                           torch::Tensor tmp, int layer_idx) {
    CHECK_INPUT(y);
    CHECK_INPUT(x);
    CHECK_INPUT(w_ptr);
    CHECK_INPUT(s);
    CHECK_INPUT(tmp);

    CHECK_DIM(2, y);
    CHECK_DIM(2, x);
    CHECK_DIM(1, w_ptr);
    CHECK_DIM(1, s);
    CHECK_DIM(1, tmp);

    int num_problems = s.size(0) - 1;
    int d_in = x.size(1);
    int d_out = y.size(1);
    CHECK_EQ(tmp.size(0), static_cast<int64_t>(sgmv_tmp_size(num_problems)));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    bool ok = DISPATCH_TORCH_DTYPE(x.scalar_type(), [&] {
      return sgmv<c_type>((c_type*)y.data_ptr(), (c_type*)x.data_ptr(),
                          (c_type**)w_ptr.data_ptr(), s.data_ptr<int32_t>(),
                          tmp.data_ptr<uint8_t>(), num_problems, d_in, d_out,
                          layer_idx, stream);
    });
    TORCH_CHECK(ok, "No suitable kernel.", " dtype=", x.scalar_type());
  }

  //====== bgmv ======

  template <typename T>
  inline bool launch_bgmv_kernel(T* Y, const T* X, const T* W,
                                const int64_t* lora_indices,
                                uint16_t in_features, uint16_t out_features,
                                int64_t batch_size, int64_t num_layers,
                                int64_t layer_idx, float scale) {
    switch (pack_u16(in_features, out_features)) {
  #define CASE_ONESIDE(_T, feat_in, feat_out)                           \
    case pack_u16(feat_in, feat_out):                                   \
      bgmv_kernel<feat_in, feat_out>(Y, X, W, lora_indices, batch_size, \
                                    num_layers, layer_idx, scale);     \
      break;
  #define CASE(_T, narrow, wide)  \
    CASE_ONESIDE(T, narrow, wide) \
    CASE_ONESIDE(T, wide, narrow)

      FOR_BGMV_WIDE_NARROW(CASE, _)
  #undef CASE
  #undef CASE_ONESIDE
      default:
        return false;
    }

    return true;
  }

  void dispatch_bgmv(torch::Tensor y, torch::Tensor x, torch::Tensor w,
                    torch::Tensor indicies, int64_t layer_idx, float scale) {
    CHECK_INPUT(y);
    CHECK_INPUT(x);
    CHECK_INPUT(w);
    CHECK_INPUT(indicies);

    CHECK_DIM(2, y);
    CHECK_DIM(2, x);
    CHECK_DIM(4, w);
    CHECK_DIM(1, indicies);

    int64_t B = x.size(0);
    int64_t h_in = x.size(1);
    int64_t h_out = y.size(1);
    int64_t num_layers = w.size(1);
    CHECK_EQ(w.size(3), h_in);
    CHECK_EQ(w.size(2), h_out);
    CHECK_EQ(indicies.size(0), x.size(0));
    CHECK_EQ(y.size(0), x.size(0));
    bool ok = false;
    if (h_in < 65536 && h_out < 65536) {
      switch (x.scalar_type()) {
        case at::ScalarType::Half:
          ok = launch_bgmv_kernel(static_cast<nv_half*>(y.data_ptr()),
                                  static_cast<nv_half*>(x.data_ptr()),
                                  static_cast<nv_half*>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out, B,
                                  num_layers, layer_idx, scale);
          break;
        case at::ScalarType::BFloat16:
          ok = launch_bgmv_kernel(static_cast<nv_bfloat16*>(y.data_ptr()),
                                  static_cast<nv_bfloat16*>(x.data_ptr()),
                                  static_cast<nv_bfloat16*>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out, B,
                                  num_layers, layer_idx, scale);
          break;
        default:
          break;
      }
    }
    TORCH_CHECK(ok, "No suitable kernel.", " h_in=", h_in, " h_out=", h_out,
                " dtype=", x.scalar_type());
  }
} // namespace punica

namespace triteia {
const int ERR_PROB_SHAPE = 1;
const int ERR_KERN_SHAPE = 2;
int triteia_cuda_bmm_2_4(const void *A, const void *B, const void *meta,
                         void *C, void *s, int prob_m, int prob_n, int prob_k,
                         void *workspace, int groupsize = -1, int dev = 0,
                         cudaStream_t stream = 0, int thread_k = -1,
                         int thread_m = -1, int sms = -1, int max_par = 16);

int triteia_cuda_sbmm_2_4(const void *A, const void *B, const void *meta,
                          void *C, void *s, const void *indices,
                          const void *starts, const void *counts, int prob_m,
                          int prob_n, int prob_k, int prob_r, void *workspace,
                          int groupsize = -1, int dev = 0,
                          cudaStream_t stream = 0, int thread_k = -1,
                          int thread_m = -1, int sms = -1, int max_par = 16);

void bmm_2_4(const torch::Tensor &A, const torch::Tensor &B,
             const torch::Tensor &meta, torch::Tensor &C,
             const torch::Tensor &s, torch::Tensor &workspace,
             int thread_k = -1, int thread_m = -1, int sms = -1,
             int max_par = 16) {
  /**
   * A:    [n, k]: n: #reqs, k: in features
   * B:    [n, k/16, 2*m]: n: #reqs, k: in features, m: out features
   * C:    [n, m]: n: #reqs, m: out features
   * s:    [n, 1, m]: n: #reqs, m: out features
   * meta: [n, k, m/16]: n: #reqs, k: in features, m: out features
   */

  int prob_n = A.size(0);
  int prob_m = C.size(2);
  int prob_k = A.size(2);

  int groupsize = (s.size(1) == 1) ? -1 : prob_k / s.size(1);
  if (groupsize != -1 && groupsize * s.size(1) != prob_k)
    AT_ERROR("k=", prob_k, " not compatible with ", s.size(0), " groups.");
  if (workspace.numel() < prob_n * prob_m / 128 * max_par)
    AT_ERROR("workspace must be of size at least ",
             prob_n * prob_m / 128 * max_par, ".");
  int dev = A.get_device();
  // print meta size
  int err = triteia_cuda_bmm_2_4(
      A.data_ptr(), B.data_ptr(), meta.data_ptr(), C.data_ptr(), s.data_ptr(),
      prob_m, prob_n, prob_k, workspace.data_ptr(), groupsize, dev,
      at::cuda::getCurrentCUDAStream(dev), thread_k, thread_m, sms, max_par);
  if (err == ERR_PROB_SHAPE) {
    AT_ERROR("Problem (m=", prob_m, ", n=", prob_n, ", k=", prob_k, ")",
             " not compatible with thread_k=", thread_k,
             ", thread_m=", thread_m, ".");
  } else if (err == ERR_KERN_SHAPE) {
    AT_ERROR("No kernel implementation for thread_k=", thread_k,
             ", thread_m=", thread_m, ", groupsize=", groupsize, ".");
  }
}

void sbmm_forloop(const torch::Tensor &A, const torch::Tensor &B,
                  const torch::Tensor &meta, torch::Tensor &C,
                  const torch::Tensor &s, const torch::Tensor &indices,
                  torch::Tensor &workspace, const torch::Tensor &starts,
                  const torch::Tensor &counts, int thread_k = -1,
                  int thread_n = -1, int sms = -1, int max_par = 8) {
  for (int i = 0; i < indices.size(0); i++) {
    int start = starts[i].item<int>();
    auto sliced_C = C.slice(0, start, start + counts[i].item<int>());
    auto my_workspace = workspace[i];
    marlin::mul_2_4(A.slice(0, start, start + counts[i].item<int>()),
                    B[indices[i]], meta[indices[i]], sliced_C, s[indices[i]],
                    my_workspace, thread_k, thread_n, sms, max_par);
  }
}

void sbmm_2_4(const torch::Tensor &A, const torch::Tensor &B,
              const torch::Tensor &meta, torch::Tensor &C,
              const torch::Tensor &s, const torch::Tensor &indices,
              torch::Tensor &workspace, const torch::Tensor &starts,
              const torch::Tensor &counts, int thread_k = -1, int thread_n = -1,
              int sms = -1, int max_par = 8) {
  int prob_n = A.size(0);
  int prob_m = C.size(1);
  int prob_k = A.size(1);
  int prob_r = indices.size(0);
  int groupsize = (s.size(1) == 1) ? -1 : prob_k / s.size(1);

  if (groupsize != -1 && groupsize * s.size(1) != prob_k)
    AT_ERROR("k=", prob_k, " not compatible with ", s.size(0), " groups.");
  if (workspace.numel() < prob_n / 128 * max_par)
    AT_ERROR("workspace must be of size at least ", prob_n / 128 * max_par,
             ".");
  int dev = A.get_device();
  int err = triteia_cuda_sbmm_2_4(
      A.data_ptr(), B.data_ptr(), meta.data_ptr(), C.data_ptr(), s.data_ptr(),
      indices.data_ptr(), starts.data_ptr(), counts.data_ptr(), prob_m, prob_n,
      prob_k, prob_r, workspace.data_ptr(), groupsize, dev,
      at::cuda::getCurrentCUDAStream(dev), thread_k, thread_n, sms, max_par);

  if (err == ERR_PROB_SHAPE) {
    AT_ERROR("Problem (m=", prob_m, ", n=", prob_n, ", k=", prob_k, ")",
             " not compatible with thread_k=", thread_k,
             ", thread_n=", thread_n, ".");
  } else if (err == ERR_KERN_SHAPE) {
    AT_ERROR("No kernel implementation for thread_k=", thread_k,
             ", thread_n=", thread_n, ", groupsize=", groupsize, ".");
  }
}
}  // namespace triteia

namespace vllm {
void rotary_embedding(torch::Tensor &positions, torch::Tensor &query, torch::Tensor &key, int64_t head_size, torch::Tensor &cos_sin_cache, bool is_neox);

void batched_rotary_embedding(torch::Tensor &positions, torch::Tensor &query,torch::Tensor &key, int64_t head_size, torch::Tensor &cos_sin_cache, bool is_neox, int64_t rot_dim, torch::Tensor &cos_sin_cache_offsets);

void silu_and_mul(torch::Tensor& out, torch::Tensor& input);

void gelu_and_mul(torch::Tensor& out, torch::Tensor& input);

void gelu_tanh_and_mul(torch::Tensor& out, torch::Tensor& input);

void fatrelu_and_mul(torch::Tensor& out, torch::Tensor& input, double threshold);

void gelu_new(torch::Tensor& out, torch::Tensor& input);

void gelu_fast(torch::Tensor& out, torch::Tensor& input);

void gelu_quick(torch::Tensor& out, torch::Tensor& input);
}  // namespace vllm

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sgmv_cutlass", &punica::dispatch_sgmv_cutlass, "");
  m.def("sgmv_cutlass_tmp_size", &sgmv_tmp_size, "");
  m.def("dispatch_bgmv", &punica::dispatch_bgmv, "dispatch_bgmv");
  m.def("mul_2_4", &marlin::mul_2_4,
        "Marlin FP16xINT4 matmul with 2:4 sparsity.");
  m.def("bmm_2_4", &triteia::bmm_2_4, "FP16xINT4 bmm with 2:4 sparsity.");
  m.def("sbmm_forloop", &triteia::sbmm_forloop,
        "FP16xINT4 sbmm with 2:4 sparsity.");
  m.def("sbmm_2_4", &triteia::sbmm_2_4, "FP16xINT4 sbmm with 2:4 sparsity.");

  m.def("rotary_embedding", &vllm::rotary_embedding, "Apply GPT-NeoX or GPT-J style rotary embedding to query and key.");
  m.def("batched_rotary_embedding", &vllm::batched_rotary_embedding, "Apply GPT-NeoX or GPT-J style rotary embedding to query and key (supports multiple loras).");
  m.def("gelu_quick", &vllm::gelu_quick, "GELU activation function.");
}
