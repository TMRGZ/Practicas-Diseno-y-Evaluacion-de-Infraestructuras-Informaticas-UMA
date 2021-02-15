#!/bin/csh -f

setenv LD_LIBRARY_PATH /usr/local/cuda/lib

make emu=1 dbg=1 verbose=1 clean
if (-f lab3.1-matrixmul.bin) then
    rm -vf lab3.1-matrixmul.bin lab3.1-matrixmul.gold
endif
echo ""
make emu=1 dbg=1 verbose=1

