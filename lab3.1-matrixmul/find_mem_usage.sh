#!/bin/sh

thisprog=$0

maincu=$1

echo "nvcc --ptxas-options=-v -I. -I../../common/inc -I/usr/local/cuda/include -DUNIX  -o $maincu.cubin -cubin $maincu"

/usr/local/cuda/bin/nvcc --ptxas-options=-v -I. -I../../common/inc -I/usr/local/cuda/include -DUNIX  -o $maincu.cubin -cubin $maincu

echo "See $maincu.cubin for register usage."
