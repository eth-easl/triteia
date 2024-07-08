#ifndef MARLIN_NM_CUDA_KERNEL_CUH_
#define MARLIN_NM_CUDA_KERNEL_CUH_

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <iostream>

#include "common/base.h"
#include "common/mem.h"
#include "common/mma.h"
#include "marlin.cuh"

namespace marlin {
// 8 warps are a good choice since every SM has 4 schedulers and having more
// than 1 warp per schedule allows some more latency hiding. At the same time,
// we want relatively few warps to have many registers per warp and small tiles.
const int THREADS = 256;
const int STAGES = 4;  // 4 pipeline stages fit into shared memory
const int SHARED_MEM =
    96 * 1024;  // max shared memory on compute capability 8.6 (< 8.0)

#define CALL_IF_2_4(THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS,         \
                    GROUP_BLOCKS)                                              \
  else if (thread_m_blocks == THREAD_M_BLOCKS &&                               \
           thread_n_blocks == THREAD_N_BLOCKS &&                               \
           thread_k_blocks == THREAD_K_BLOCKS &&                               \
           group_blocks == GROUP_BLOCKS) {                                     \
    cudaFuncSetAttribute(Marlin_2_4<THREADS, THREAD_N_BLOCKS, THREAD_M_BLOCKS, \
                                    THREAD_K_BLOCKS, STAGES, GROUP_BLOCKS>,    \
                         cudaFuncAttributeMaxDynamicSharedMemorySize,          \
                         SHARED_MEM);                                          \
    Marlin_2_4<THREADS, THREAD_N_BLOCKS, THREAD_M_BLOCKS, THREAD_K_BLOCKS,     \
               STAGES, GROUP_BLOCKS><<<blocks, THREADS, SHARED_MEM, stream>>>( \
        A_ptr, B_ptr, meta_ptr, C_ptr, s_ptr, prob_n, prob_m, prob_k, locks);  \
  }

const int ERR_PROB_SHAPE = 1;
const int ERR_KERN_SHAPE = 2;

int marlin_cuda_2_4(const void *A, const void *B, const void *meta, void *C,
                    void *s, int prob_m, int prob_n, int prob_k,
                    void *workspace, int groupsize = -1, int dev = 0,
                    cudaStream_t stream = 0, int thread_k = -1,
                    int thread_m = -1, int sms = -1, int max_par = 16) {
  int tot_n = prob_n;
  int tot_n_blocks = ceildiv(tot_n, 16);
  int pad = 16 * tot_n_blocks - tot_n;

  if (sms == -1)
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);

  if (thread_k == -1 || thread_m == -1) {
    if (prob_n <= 16) {
      // For small batchizes, better partioning is slightly more important than
      // better compute utilization
      thread_k = 128;
      thread_m = 128;
    } else {
      thread_k = 64;
      thread_m = 256;
    }
  }

  int thread_k_blocks = thread_k / 32;  // 2:4 version with m16n8k32 instruction
  int thread_m_blocks = thread_m / 16;
  int group_blocks = (groupsize == -1) ? -1 : groupsize / 16;
  int blocks = sms;
  if (prob_m % thread_m != 0 || prob_k % thread_k != 0 ||
      (group_blocks != -1 && (prob_k / 2) % group_blocks != 0))
    return ERR_PROB_SHAPE;
  if (prob_m == 0 || prob_n == 0 || prob_k == 0) return 0;
  const int4 *A_ptr = (const int4 *)A;
  const int4 *B_ptr = (const int4 *)B;
  const int4 *meta_ptr = (const int4 *)meta;
  int4 *C_ptr = (int4 *)C;
  const int4 *s_ptr = (const int4 *)s;

  int cols = prob_m / thread_m;
  int *locks = (int *)workspace;
  int ret = 0;
  for (int i = 0; i < tot_n_blocks; i += 4) {
    int thread_n_blocks = tot_n_blocks - i;
    prob_n = tot_n - 16 * i;
    int par = 1;
    if (thread_n_blocks > 4) {
      // Note that parallel > 1 currently only works for inputs without any
      // padding
      par = (16 * thread_n_blocks - pad) / 64;
      if (par > max_par) par = max_par;
      prob_n = 64 * par;
      i += 4 * (par - 1);
      thread_n_blocks = 4;
    }

    // For compilation speed, we only define the kernel configurations that have
    // seemed useful (in terms of performance) in our testing, however many more
    // are, in principle, possible.
    if (false) {
    }  //         BMxBNxBK,   group
    CALL_IF_2_4(8, 1, 4, -1)   // e.g., 16x128x128
    CALL_IF_2_4(8, 1, 4, 4)    // e.g., 16x128x128, 64
    CALL_IF_2_4(16, 1, 2, -1)  // e.g., 16x256x64
    CALL_IF_2_4(16, 1, 2, 4)   // e.g., 16x256x64,  64
    CALL_IF_2_4(16, 2, 2, -1)  // e.g.. 32x256x64
    CALL_IF_2_4(16, 2, 2, 4)
    CALL_IF_2_4(16, 3, 2, -1)
    CALL_IF_2_4(16, 3, 2, 4)
    CALL_IF_2_4(16, 4, 2, -1)
    CALL_IF_2_4(16, 4, 2, 4)
    else ret = ERR_KERN_SHAPE;

    A_ptr += 16 * thread_n_blocks * (prob_k / 8) * par;
    C_ptr += 16 * thread_n_blocks * (prob_m / 8) * par;
  }

  return ret;
}
#endif
}