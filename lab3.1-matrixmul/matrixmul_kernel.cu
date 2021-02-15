/* Matrix multiplication: C = A * B.
 * Device code.
 
 * Copyright by Nvidia. Used and adapted by Manuel Ujaldon 
 * during his seminars as Nvidia CUDA Fellow.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "matrixmul.h"

#define CHECK_BANK_CONFLICTS 0
#if CHECK_BANK_CONFLICTS
# define AS(i, j) CUT_BANK_CHECKER(((float *)&As[0][0]), (BLOCK_SIZE * i + j))
# define BS(i, j) CUT_BANK_CHECKER(((float *)&Bs[0][0]), (BLOCK_SIZE * i + j))
#else
# define AS(i, j) As[i][j]
# define BS(i, j) Bs[i][j]
#endif

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////

    __global__ void
matrixMul(float *C, float *A, float *B, int Aw, int Bw)
{
    // Block index
    #define bx blockIdx.x
    #define by blockIdx.y

    // Thread index
    #define tx threadIdx.x
    #define ty threadIdx.y
    // Index of the first element of A loaded by this thread by the block
    int aEnd = Aw * ty;
    aEnd += tx;
    int a = Aw * BLOCK_SIZE * by + aEnd;
    int b = BLOCK_SIZE * bx;

    #ifdef SPILL
    // Create a shared-memory buffer to spill a register value
    // into shared memory, hopefully reducing the total required
    // register count.
    __shared__ int c[BLOCK_SIZE][BLOCK_SIZE];
    c[tx][ty] = a + b;
    #else
    int c =  a + b;
    #endif

    // Index of the first sub-matrix of B processed by the block
    b = b + aEnd;

    // Index of the last sub-matrix of A processed by the block
    aEnd = a + Aw;

    // Step size used to iterate through the sub-matrices of A
    #define aStep BLOCK_SIZE

    // Step size used to iterate through the sub-matrices of B
    #define bStep BLOCK_SIZE * Bw

    // Initialize result(s) to 0.
    float Csub = 0;

    // Initial prefetch.  Issues loads to main memory and store
    // in temporary variables which will later be stored to shared memory
    #ifdef PREFETCH
    float fa = A[a];
    float fb = B[b];
    #endif

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    while (a < aEnd) {
        #if UNROLL != 16
        int i;
        #endif

        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrices of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        #ifdef PREFETCH
        // If performing prefetching, the values are already loaded
        // from memory, and the temporary variables holding the loaded
        // values are stored to shared memory.
        As[ty][tx] = fa;
        Bs[ty][tx] = fb;
        #else
        // Load the matrices from device memory
        // directly to shared memory
        As[ty][tx] = A[a];
        Bs[ty][tx] = B[b];
        #endif
               // Update for next loop
        a += aStep;
        b += bStep;

        // Synchronize to make sure the shared memory
        // tiles are ready
        __syncthreads();

        #ifdef PREFETCH
        // If prefetching, issue the loads for the next tiles preemptively.
        // The loads will complete and be stored into these temporary
        // variables while the current shared memory tiles
        // are being operated on.
        fa = A[a];
        fb = B[b];
        #endif

        // Multiply the two matrices together.
        #if UNROLL == 16
        // Completely unrolled: no actual loop structure
        Csub += As[ty][0] *Bs[0][tx];
        Csub += As[ty][1] *Bs[1][tx];
        Csub += As[ty][2] *Bs[2][tx];
        Csub += As[ty][3] *Bs[3][tx];
        Csub += As[ty][4] *Bs[4][tx];
        Csub += As[ty][5] *Bs[5][tx];
        Csub += As[ty][6] *Bs[6][tx];
        Csub += As[ty][7] *Bs[7][tx];
          # if (BLOCK_SIZE == 12 || BLOCK_SIZE == 16 || BLOCK_SIZE == 32)
          Csub += As[ty][8] *Bs[8][tx];
          Csub += As[ty][9] *Bs[9][tx];
          Csub += As[ty][10] *Bs[10][tx];
          Csub += As[ty][11] *Bs[11][tx];
            # if (BLOCK_SIZE == 16 || BLOCK_SIZE == 32)
            Csub += As[ty][12] *Bs[12][tx];
            Csub += As[ty][13] *Bs[13][tx];
            Csub += As[ty][14] *Bs[14][tx];
            Csub += As[ty][15] *Bs[15][tx];
              #  if BLOCK_SIZE == 32
              Csub += As[ty][16] *Bs[16][tx];
              Csub += As[ty][17] *Bs[17][tx];
              Csub += As[ty][18] *Bs[18][tx];
              Csub += As[ty][19] *Bs[19][tx];
              Csub += As[ty][20] *Bs[20][tx];
              Csub += As[ty][21] *Bs[21][tx];
              Csub += As[ty][22] *Bs[22][tx];
              Csub += As[ty][23] *Bs[23][tx];
              Csub += As[ty][24] *Bs[24][tx];
              Csub += As[ty][25] *Bs[25][tx];
              Csub += As[ty][26] *Bs[26][tx];
              Csub += As[ty][27] *Bs[27][tx];
              Csub += As[ty][28] *Bs[28][tx];
              Csub += As[ty][29] *Bs[29][tx];
              Csub += As[ty][30] *Bs[30][tx];
              Csub += As[ty][31] *Bs[31][tx];
              # endif // BLOCK_SIZE == 32
            # endif // BLOCK_SIZE == 16 || 32
          # endif // BLOCK_SIZE == 12 || 16 || 32

        #elif UNROLL == 4
        // Loop unrolled three times
        for (i = 0; i < BLOCK_SIZE; i += 4) {
            Csub += As[ty][i] *Bs[i][tx];
            Csub += As[ty][i + 1] *Bs[i + 1][tx];
            Csub += As[ty][i + 2] *Bs[i + 2][tx];
            Csub += As[ty][i + 3] *Bs[i + 3][tx];
        }

        #elif UNROLL == 2
        // Loop unrolled once
        for (i = 0; i < BLOCK_SIZE; i += 2) {
            Csub += As[ty][i] *Bs[i][tx];
            Csub += As[ty][i + 1] *Bs[i + 1][tx];
        }

        #elif UNROLL == 0
        // Regular loop, no unrolling
        for (i = 0; i < BLOCK_SIZE; i++) {
            Csub += As[ty][i] *Bs[i][tx];
        }
        #endif // UNROLL

        // Synchronize to make sure that the preceding
        // computation is done before overwriting new
        // shared memory sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    #ifdef SPILL
    // If we spilled the output index at the beginning, load it back
    // from the shared memory array.
    // Output the result(s) for each thread.
    C[c[tx][ty]] = Csub;

    #else
    // Output the final result(s) for each thread.
    C[c] = Csub;
    #endif
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_

