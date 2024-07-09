#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <torch/python.h>

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
    marlin::mul_2_4(A.slice(0, start, start + counts[i].item<int>()), B[indices[i]],
            meta[indices[i]], sliced_C, s[indices[i]], my_workspace, thread_k,
            thread_n, sms, max_par);
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mul_2_4", &marlin::mul_2_4,
        "Marlin FP16xINT4 matmul with 2:4 sparsity.");
  m.def("bmm_2_4", &triteia::bmm_2_4, "FP16xINT4 bmm with 2:4 sparsity.");
  m.def("sbmm_forloop", &triteia::sbmm_forloop,
        "FP16xINT4 sbmm with 2:4 sparsity.");
  m.def("sbmm_2_4", &triteia::sbmm_2_4, "FP16xINT4 sbmm with 2:4 sparsity.");
}