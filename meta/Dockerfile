FROM nvcr.io/nvidia/pytorch:24.07-py3

ENV TORCH_CUDA_ARCH_LIST=Ampere
ENV TRITEIA_COMPUTE_CAP=86
WORKDIR /triteia
COPY . .
RUN pip install flash-attn --no-build-isolation
RUN pip install -r requirements.txt
RUN pip install -e .
