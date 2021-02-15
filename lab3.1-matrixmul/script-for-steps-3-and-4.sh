#! /bin/sh
make clean
make
CUDA_PROFILE=1
CUDA_PROFILE_CONFIG=./profile_config
export CUDA_PROFILE
export CUDA_PROFILE_CONFIG
./matrixmul 512
cat cuda_profile_0.log
./find_mem_usage.sh matrixmul.cu
