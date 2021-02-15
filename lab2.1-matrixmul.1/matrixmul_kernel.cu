/* Matrix multiplication: P = M * N.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "matrixmul.h"


////////////////////////////////////////////////////////////////////////////////
//! Matrix multiplication on the device: P = M * N
//! Mw is M's width and Nw is N's width
////////////////////////////////////////////////////////////////////////////////
__global__ void matrixMul( float* P, float* M, float* N, int Mw, int Nw)
{
    int bx = blockIdx.x;     int by = blockIdx.y;
    int tx = threadIdx.x;    int ty = threadIdx.y;
    __shared__ float Ms[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Ns[BLOCK_SIZE][BLOCK_SIZE];

    // ===================================================================
    // Code segment 1
    // Determine the update values for the tile indices in the loop
    // ===================================================================

    int mBegin = Mw * BLOCK_SIZE * by; // IndexY - ty
    int mEnd   = mBegin + Mw - 1;
    int mStep  = BLOCK_SIZE;
    int nBegin = BLOCK_SIZE * bx; // IndexX - tx
    int nStep  = BLOCK_SIZE * Nw;
    float Psub = 0.0f;

    // ===================================================================
    // Code segment 2
    // Do matrix-matrix multiplication inside a tile
    // ===================================================================

    for (int m = mBegin, n = nBegin; m <= mEnd; m += mStep, n += nStep) {

        // Load a tile from M and N into the shared memory arrays
        Ms[ty][tx] = M[m + Mw * ty + tx];
        Ns[ty][tx] = N[n + Nw * ty + tx];

        // Synchronize the threads
        __syncthreads();

        // Multiply the two tiles together, each thread accumulating
        // the partial sum of a single dot product.
        for (int i = 0; i < BLOCK_SIZE; i++) {
            Psub += Ms[ty][i] * Ns[i][tx];
        }

        // Synchronize again.
        __syncthreads();
    }

    // ===================================================================
    // Code segment 3
    // Store the data back to global memory
    // ===================================================================

    int p = Nw * BLOCK_SIZE * by + BLOCK_SIZE * bx; // mBegin + nBegin
    P[p + Nw * ty + tx] = Psub;
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
