import sys
from functools import lru_cache
from subprocess import DEVNULL, call
from setuptools import setup, find_packages

import torch
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


@lru_cache(None)
def cuda_toolkit_available():
  try:
    call(["nvcc"], stdout = DEVNULL, stderr = DEVNULL)
    print('CUDA toolkit available - building CUDA extension')
    return True
  except FileNotFoundError:
    print('CUDA toolkit not available - skipping CUDA extension')
    return False

def compile_args():
  args = ["-fopenmp", "-ffast-math"]
  if sys.platform == "darwin":
    args = ["-Xpreprocessor", *args]
  return args

def ext_modules():
  if not cuda_toolkit_available():
    print('CUDA toolkit not available - skipping CUDA extension')
    return []

  print('CUDA toolkit available - building CUDA extension')
  return [
    CUDAExtension(
      "flash_cosine_sim_attention_cuda",
      sources = ["mem_pytorch/cosine/flash_cosine_sim_attention_cuda.cu"]
    )
  ]

setup(
  name = 'mem_pytorch',
  packages = find_packages(exclude=[]),
  version = '0.1.0',
  license='MIT',
  description = 'A PyTorch implementation of the MEM model',
  long_description_content_type = 'text/markdown',
  author = 'Phil Wang',
  author_email = 'robert@opentensor.ai',
  url = 'https://github.com/opentensor/mem-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'attention-mechanism'
  ],
  install_requires=[
    'einops>=0.4.1',
  ],
  setup_requires=[
    'pytest-runner',
  ],
  tests_require=[
    'pytest'
  ],
  ext_modules = ext_modules(),
  cmdclass = {"build_ext": BuildExtension},
  include_package_data = True,
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
