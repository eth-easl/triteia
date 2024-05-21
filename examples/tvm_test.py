import numpy as np
import tvm
import tvm.testing
from tvm import te, auto_scheduler, topi

@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def logit_forward(N, L, D, C, S, dtype):
    A = te.placeholder((N, C), name="A", dtype=dtype)
    I = te.placeholder((N, L), name='I', dtype='int32')
    K = te.placeholder((D, S, C), name="K", dtype=dtype)
    B = te.placeholder((D, S,), name="B", dtype=dtype)
    
    # this step is to make sure the index do not go over the bound
    # remember remove related code
    I_1 = topi.abs(I)
    I_2 = topi.mod(I_1, D)
    I_3 = topi.mod(I_1, D)
    t = te.reduce_axis((0, C), name="t")
    key = te.compute(
        (N, L, S),
        lambda i, j, k: te.sum(A[i, t] * K[I_2[i, j], k, t], axis=t),
        name="compute_key",
        attrs={"layout_free_placeholders": [K]},  # enable automatic layout transform for tensor B
    )
    
    key_bias = te.compute(
        (N, L, S),
        lambda i, j, k: key[i, j, k] + B[I_3[i, j], k],
        name="add_bias",
    )

    return [A, I, K, B, key_bias]

gpu_target = tvm.target.Target("cuda")

N = 5
L = 3
D = 6
C = 3
S = 4

def check_correctness():
    i_feat = np.random.rand(N, C).astype("float32")
    index = np.random.randint(0, D, (N, L)).astype("int32")
    linear = np.random.rand(D, S, C).astype("float32")
    bias = np.random.rand(D, S).astype("float32")
    
    act = np.einsum('nc,dsc->nds', i_feat, linear)
    act = act + bias[None]
    result = np.take_along_axis(act, index[:,:,None], axis=1)
    
    nodes = logit_forward(N, L, D, C, S, "float32")
    s = te.create_schedule(nodes[-1].op)
    func = tvm.build(s, nodes, gpu_target, name="act")
    
    dev = tvm.device(gpu_target.kind.name, 0)
    i_feat = tvm.nd.array(i_feat, dev)
    index = tvm.nd.array(index, dev)
    linear = tvm.nd.array(linear, dev)
    bias = tvm.nd.array(bias, dev)
    out = tvm.nd.array(np.zeros_like(result), dev)
    
    func(i_feat, index, linear, bias, out)
    
    tvm.testing.assert_allclose(out.numpy(), result)

check_correctness()

task = tvm.auto_scheduler.SearchTask(func=logit_forward, args=(N, L, D, C, S, "float32"), target=gpu_target)

# Inspect the computational graph
print("Computational DAG:")
print(task.compute_dag)

log_file = "ops/logit_forward.json"
measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=10,
    runner=measure_ctx.runner,
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    verbose=2,
)

# Run auto-tuning (search)
task.tune(tune_option)
# Apply the best schedule
sch, args = task.apply_best(log_file)

print("Lowered TIR:")
print(tvm.lower(sch, args, simple_mode=True))