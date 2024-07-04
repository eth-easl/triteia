#pragma once
#include "cutlass/gemm/kernel/grouped_problem_visitor.h"

namespace cutlass {
namespace gemm {
namespace kernel {

template <typename ProblemSizeHelper, typename ThreadblockShape_>
struct SBMMProblemVisitor {
  using ThreadblockShape = ThreadblockShape_;
  int32_t problem_idx;
  int32_t problem_start;

  CUTLASS_DEVICE
  ProblemInfo()
      : problem_idx(kNoPrefetchEntry), problem_start(kNoPrefetchEntry) {}

  CUTLASS_DEVICE
  ProblemInfo(int32_t problem_idx_, int32_t problem_start_)
      : problem_idx(problem_idx_), problem_start(problem_start_) {}
};

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass