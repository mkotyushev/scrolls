# Introduction

This repository contains deep learning development environment for scrolls project.

# Installation

### APEX
1. Clone the repository: `git clone https://github.com/NVIDIA/apex`
2. Change directory: `cd apex`
3. Checkout: `git checkout 82ee367f3da74b4cd62a1fb47aa9806f0f47b58b`
4. Build & install, providing path to proper gcc and g++: `CC='/usr/bin/gcc-9' CXX='/usr/bin/g++-9' pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./`