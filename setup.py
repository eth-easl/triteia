"""Python setup.py for triteia package"""

import io
import os
from setuptools import find_packages, setup
from torch.utils import cpp_extension


def get_compute_capability():
    try:
        compute_cap = (
            os.popen("nvidia-smi --query-gpu=compute_cap --format=csv,noheader")
            .read()
            .strip()
            .split("\n")[0]
        )
        major, minor = compute_cap.split(".")
        return f"{major}{minor}"
    except Exception as e:
        print(f"Failed to detect compute capability: {e}")
        return None


def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    >>> read("triteia", "VERSION")
    '0.1.0'
    >>> read("README.md")
    ...
    """
    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    return [
        line.strip()
        for line in read(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]


compute_cap = get_compute_capability()
if compute_cap is None:
    raise ValueError("Failed to detect compute capability")

setup(
    name="triteia",
    version=read("triteia", "VERSION"),
    description="Fast GPU Kernels",
    url="https://github.com/eth-easl/triteia/",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="Xiaozhe Yao",
    packages=find_packages(exclude=["tests", ".github", "benchmarks", "docs"]),
    install_requires=read_requirements("requirements.txt"),
    extras_require={"test": read_requirements("requirements-dev.txt")},
    ext_modules=[
        cpp_extension.CUDAExtension(
            "triteia_cuda",
            [
                "triteia/csrc/ops/ops.cpp",
                "triteia/csrc/ops/marlin_nm.cu",
                "triteia/csrc/ops/triteia_nm_bmm.cu",
                "triteia/csrc/ops/triteia_nm_sbmm.cu",
            ],
            dlink=True,
            extra_compile_args={
                "nvcc": [
                    "-O3",
                    f"-arch=sm_{compute_cap}",
                    "--ptxas-options=-v",
                    "-dc",
                    "-lineinfo",
                ]
            },
            extra_link_args=["-lcudadevrt", "-lcudart"],
        ),
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
