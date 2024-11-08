# Installation

### Development
```
mamba install cuda=12.4.1=0 -c nvidia/label/cuda-12.4.1 -c conda-forge
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

git submodule update --init --recursive
```