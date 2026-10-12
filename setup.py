import os
import torch
import glob

from setuptools import find_packages, setup

from torch.utils.cpp_extension import (
    CUDAExtension,
    BuildExtension,
)

VERSION = "0.0.1"

library_name = "refiner_C"

if torch.__version__ >= "2.6.0":
    py_limited_api = True
else:
    py_limited_api = False

def get_cuda_extention():
    # debug_mode = os.getenv("DEBUG", "0") == "1"
    debug_mode = True
    if debug_mode:
        print("Compiling in debug mode")

    extra_link_args = []
    extra_compile_args = {
        "cxx": [
            "-O3" if not debug_mode else "-O0",
            "-fdiagnostics-color=always",
            "-DPy_LIMITED_API=0x03090000",  # min CPython version 3.9
        ],
        "nvcc": [
            "-O3" if not debug_mode else "-O0",
        ],
    }
    if debug_mode:
        extra_compile_args["cxx"].append("-g")
        extra_compile_args["nvcc"].append("-g")
        extra_link_args.extend(["-O0", "-g"])

    extensions_dir = os.path.join("refiner_C", "src")
    sources = list(glob.glob(os.path.join(extensions_dir, "*.cpp")))
    cuda_sources = list(glob.glob(os.path.join(extensions_dir, "*.cu")))
    sources += cuda_sources

    return CUDAExtension(
        f"refiner_C._C",
        sources,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        py_limited_api=py_limited_api,
    )

setup(
    name="ReFineR",
    version=VERSION,
    author="Ugo Insalaco",
    author_email="ugo.insalaco@minesparis.psl.eu",
    description="Official implementation of the ReFineR algorithm.",
    long_description_content_type="text/markdown",
    packages = ["refiner"] + find_packages(),
    package_dir={'refiner':'refiner'},
    ext_modules = [get_cuda_extention()],
    python_requires=">=3.10",
    install_requires=[],
    classifiers=["Operating System :: OS Independent"],
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp39"}} if py_limited_api else {},
)