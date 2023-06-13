# Introduction

This repository contains deep learning development environment for scrolls project.

# Installation

### mmcv & mmsegmentation
1. mmcv: `pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html`
2. mmsegmentation & deps (except apex, which needs to be build from source): `pip install xformers==0.0.20 einops==0.6.1 mmsegmentation==1.0.0`
3. apex: `cd lib && git clone https://github.com/NVIDIA/apex && cd apex && CC='/usr/bin/gcc-9' CXX='/usr/bin/g++-9' pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./`
