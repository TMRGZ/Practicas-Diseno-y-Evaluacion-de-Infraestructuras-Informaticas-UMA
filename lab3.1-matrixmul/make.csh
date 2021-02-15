#!/bin/csh -f

setenv LD_LIBRARY_PATH /usr/local/cuda/lib

make verbose=1 clean
if (-f lab3.1-matrixmul.bin) then
    rm -vf lab3.1-matrixmul.bin lab3.1-matrixmul.gold
endif
echo ""
make verbose=1

