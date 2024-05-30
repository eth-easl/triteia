import io
import os
from setuptools import find_packages, setup
from torch.utils import cpp_extension


def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    >>> read("dstool", "VERSION")
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


setup(
    name="triteia",
    version=read("triteia", "VERSION"),
    description="Useful Kernels",
    url="https://github.com/eth-easl/triteia/",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="xzyaoi",
    packages=find_packages(exclude=["tests", ".github"]),
    install_requires=read_requirements("requirements.txt"),
    entry_points={"dstool": ["dstool = dstool.__main__:main"]},
    extras_require={"test": read_requirements("requirements-test.txt")},
    ext_modules=[
        cpp_extension.CUDAExtension(
            "marlin_cuda",
            [
                "triteia/csrc/marlin/marlin_cuda.cpp",
                "triteia/csrc/marlin/marlin_cuda_kernel.cu",
                "triteia/csrc/marlin/marlin_cuda_kernel_nm.cu",
            ],
            extra_compile_args={
                "nvcc": [
                    "-arch=sm_86", "--ptxas-options=-v", "-lineinfo"
                ]
            },
        ),
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
