docker run --rm -v $(pwd):/work -w /work --env MAX_JOBS=64 --env TORCH_CUDA_ARCH_LIST="8.0 8.6 9.0" nvcr.io/nvidia/pytorch:24.09-py3 /bin/bash -c "pip install build && python -m build --no-isolation"
