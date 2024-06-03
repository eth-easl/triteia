/*
 * Copyright (C) 2024 Roberto Lopez Castro (roberto.lopez.castro@udc.es). All
 * Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <ATen/Functions.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <torch/all.h>
#include <torch/python.h>

#include <execution>
#include <iostream>
#include <numeric>
using namespace torch::indexing;

namespace marlin {

int marlin_cuda_2_4(const void *A, const void *B, const void *meta, void *C,
                    void *s, int prob_m, int prob_n, int prob_k,
                    void *workspace, int groupsize = -1, int dev = 0,
                    cudaStream_t stream = 0, int thread_k = -1,
                    int thread_m = -1, int sms = -1, int max_par = 16);

int marlin_cuda(const void *A, const void *B, void *C, void *s, int prob_m,
                int prob_n, int prob_k, void *workspace, int groupsize = -1,
                int dev = 0, cudaStream_t stream = 0, int thread_k = -1,
                int thread_n = -1, int sms = -1, int max_par = 16);

int ibmm_cuda_2_4(const void *A, const void *B, void *C, void *s,void *indices, int prob_m,
                int prob_n, int prob_k, void *workspace, int groupsize = -1,
                int dev = 0, cudaStream_t stream = 0, int thread_k = -1,
                int thread_n = -1, int sms = -1, int max_par = 16);

const int ERR_PROB_SHAPE = 1;
const int ERR_KERN_SHAPE = 2;

void mul(const torch::Tensor &A, const torch::Tensor &B, torch::Tensor &C,
         const torch::Tensor &s, torch::Tensor &workspace, int thread_k = -1,
         int thread_n = -1, int sms = -1, int max_par = 8) {
  int prob_m = A.size(0);
  int prob_n = C.size(1);
  int prob_k = A.size(1);
  int groupsize = (s.size(0) == 1) ? -1 : prob_k / s.size(0);
  if (groupsize != -1 && groupsize * s.size(0) != prob_k)
    AT_ERROR("k=", prob_k, " not compatible with ", s.size(0), " groups.");
  if (workspace.numel() < prob_n / 128 * max_par)
    AT_ERROR("workspace must be of size at least ", prob_n / 128 * max_par,
             ".");
  int dev = A.get_device();
  int err = marlin_cuda(A.data_ptr(), B.data_ptr(), C.data_ptr(), s.data_ptr(),
                        prob_m, prob_n, prob_k, workspace.data_ptr(), groupsize,
                        dev, at::cuda::getCurrentCUDAStream(dev), thread_k,
                        thread_n, sms, max_par);

  if (err == ERR_PROB_SHAPE) {
    AT_ERROR("Problem (m=", prob_m, ", n=", prob_n, ", k=", prob_k, ")",
             " not compatible with thread_k=", thread_k,
             ", thread_n=", thread_n, ".");
  } else if (err == ERR_KERN_SHAPE) {
    AT_ERROR("No kernel implementation for thread_k=", thread_k,
             ", thread_n=", thread_n, ", groupsize=", groupsize, ".");
  }
}

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
  at::cuda::CUDAStream stream = at::cuda::getStreamFromPool(0, dev);
  {
    int err = marlin_cuda_2_4(A.data_ptr(), B.data_ptr(), meta.data_ptr(),
                              C.data_ptr(), s.data_ptr(), prob_m, prob_n,
                              prob_k, workspace.data_ptr(), groupsize, dev,
                              stream, thread_k, thread_m, sms, max_par);
    if (err == ERR_PROB_SHAPE) {
      AT_ERROR("Problem (m=", prob_m, ", n=", prob_n, ", k=", prob_k, ")",
               " not compatible with thread_k=", thread_k,
               ", thread_m=", thread_m, ".");
    } else if (err == ERR_KERN_SHAPE) {
      AT_ERROR("No kernel implementation for thread_k=", thread_k,
               ", thread_m=", thread_m, ", groupsize=", groupsize, ".");
    }
  }
}

void mul_stream(const torch::Tensor &A, const torch::Tensor &B,
                const torch::Tensor &meta, torch::Tensor &C,
                const torch::Tensor &s, const torch::Tensor &indices,
                torch::Tensor &workspace, const torch::Tensor &starts,
                const torch::Tensor &counts, int thread_k = -1,
                int thread_n = -1, int sms = -1, int max_par = 8) {
  for (int i = 0; i < indices.size(0); i++) {
    int start = starts[i].item<int>();
    auto sliced_C = C.slice(0, start, start + counts[i].item<int>());
    auto my_workspace = workspace[i];
    mul_2_4(A.slice(0, start, start + counts[i].item<int>()), B[indices[i]],
            meta[indices[i]], sliced_C, s[indices[i]], my_workspace, thread_k,
            thread_n, sms, max_par);
  }
}

void mul_stream_parallel(const torch::Tensor &A, const torch::Tensor &B,
                         const torch::Tensor &meta, torch::Tensor &C,
                         const torch::Tensor &s, const torch::Tensor &indices,
                         torch::Tensor &workspace, const torch::Tensor &starts,
                         const torch::Tensor &counts, int thread_k = -1,
                         int thread_n = -1, int sms = -1, int max_par = 8) {
  std::vector<std::thread> mul_threads(indices.size(0));
  for (int i = 0; i < indices.size(0); ++i) {
    mul_threads[i] =
        std::thread([i, &A, &B, &meta, &C, &s, &indices, &workspace, &starts,
                     &counts, thread_k, thread_n, sms, max_par]() {
          int  start = starts[i].item<int>();
          auto sliced_C = C.slice(0, start, start + counts[i].item<int>());
          auto my_workspace = workspace[i];
          mul_2_4(A.slice(0, start, start + counts[i].item<int>()),
                  B[indices[i]], meta[indices[i]], sliced_C, s[indices[i]],
                  my_workspace, thread_k, thread_n, sms, max_par);
        });
  }
  std::for_each(mul_threads.begin(), mul_threads.end(),
                std::mem_fn(&std::thread::join));
}

void ibmm(const torch::Tensor &A, const torch::Tensor &B, torch::Tensor &C,
         const torch::Tensor &s, const torch::Tensor &indices, torch::Tensor &workspace, 
         int thread_k = -1, int thread_n = -1, int sms = -1, int max_par = 8) {
  /*
  A: [m, k], m: # reqs, k: in features
  B: [r, n, k], r: # models, n: out features, k: in features
  C: [m, n], m: # reqs, n: out features
  s: [r, n/g (usually 1), k], r: # models, n: out features, k: in features
  indices: [m], m: # reqs, ranges from 0 to r-1
  */
  
  int prob_m = A.size(0);
  int prob_n = C.size(1);
  int prob_k = A.size(1);
  int groupsize = (s.size(0) == 1) ? -1 : prob_k / s.size(0);
  if (groupsize != -1 && groupsize * s.size(0) != prob_k)
    AT_ERROR("k=", prob_k, " not compatible with ", s.size(0), " groups.");
  if (workspace.numel() < prob_n / 128 * max_par)
    AT_ERROR("workspace must be of size at least ", prob_n / 128 * max_par,
             ".");
  int dev = A.get_device();

  int err = ibmm_cuda_2_4(
    A.data_ptr(), B.data_ptr(), C.data_ptr(), s.data_ptr(), indices.data_ptr(),
    prob_m, prob_n, prob_k, workspace.data_ptr(), groupsize,
    dev, at::cuda::getCurrentCUDAStream(dev), thread_k,
    thread_n, sms, max_par
  );
  
  if (err == ERR_PROB_SHAPE) {
    AT_ERROR("Problem (m=", prob_m, ", n=", prob_n, ", k=", prob_k, ")",
             " not compatible with thread_k=", thread_k,
             ", thread_n=", thread_n, ".");
  } else if (err == ERR_KERN_SHAPE) {
    AT_ERROR("No kernel implementation for thread_k=", thread_k,
             ", thread_n=", thread_n, ".");
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mul", &mul, "Marlin FP16xINT4 matmul.");
  m.def("mul_2_4", &mul_2_4, "Marlin FP16xINT4 matmul with 2:4 sparsity.");
  m.def("mul_stream", &mul_stream, "Marlin FP16xINT4 matmul with stream.");
  m.def("mul_stream_parallel", &mul_stream_parallel,
        "Marlin FP16xINT4 matmul with stream.");
  m.def("ibmm", &ibmm,
        "Marlin FP16xINT4 matmul with stream.");
}
}  // namespace marlin