import torch
from timeit import default_timer as timer
total_num = 16
x = torch.rand((total_num, 1,1024), device='cuda')
w = torch.rand((1024, 1024), device='cuda')
y = torch
streams = [torch.cuda.Stream() for _ in range(total_num)]

torch.cuda.synchronize()
start = timer()
for i in range(total_num):
    with torch.cuda.stream(streams[i]):
        y = torch.matmul(x[i], w.T)

torch.cuda.synchronize()
end = timer()


torch.cuda.synchronize()
start = timer()
for i in range(total_num):
    y = torch.matmul(x[i], w.T)
torch.cuda.synchronize()
end = timer()


torch.cuda.synchronize()
torch.cuda.nvtx.range_push("With stream")
start = timer()
for i in range(total_num):
    with torch.cuda.stream(streams[i]):
        y = torch.matmul(x[i], w.T)
torch.cuda.synchronize()
end = timer()
torch.cuda.nvtx.range_pop()
print(f"With stream: {end-start}")


torch.cuda.synchronize()
torch.cuda.nvtx.range_push("Without stream")
start = timer()
for i in range(total_num):
    y = torch.matmul(x[i], w.T)
torch.cuda.synchronize()
end = timer()
torch.cuda.nvtx.range_pop()

print(f"Without stream: {end-start}")
