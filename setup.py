from setuptools import setup, find_packages

setup(
  name = 'mem-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.1.0',
  license='MIT',
  description = 'Memory Efficient Attention - Pytorch',
  long_description_content_type = 'text/markdown',
  author = 'Phil Wang / jouee / carro',
  author_email = 'robert@opentensor.ai',
  url = 'https://github.com/opentensor/mem-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'attention-mechanism'
  ],
  install_requires=[
    'einops>=0.4.1',
    'torch>=1.6',
  ],
  setup_requires=[
    'pytest-runner',
  ],
  tests_require=[
    'pytest'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
