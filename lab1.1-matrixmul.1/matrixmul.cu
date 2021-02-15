/* Matrix multiplication: P = M * N.
 * Host code.
 */

// includes, system
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, kernels
#include "matrixmul_kernel.cu"

#include "assist.h"

#define ERROR_CHECK { cudaError_t err; \
  if ((err = cudaGetLastError()) != cudaSuccess) { \
    printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__);}}

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char** argv);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
    runTest(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char** argv)
{
    bool if_quiet = true;
    int i, j;
    char *matrix_id = NULL, *input_fn = NULL, *gold_fn = NULL;
    float * deviceM = NULL, * deviceN = NULL, * deviceP = NULL;
    int Mw = 0, Mh = 0, Nw = 0, Nh = 0, Pw = 0, Ph = 0;
    int block_size = 0;

    cudaEvent_t start, stop;
    float timer_compute, timer_memory_in, timer_memory_out;
// Create CUDA events for measuring execution times
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    if (argc == 2) {
        matrix_id = strdup(argv[1]);

    } else {
        fprintf(stderr, "Error: Wrong input parameter numbers.\n");
        fprintf(stderr, "Usage:\n"
                        "$> ./lab1.1-matrixmul <8, 128, 512, 3072, 4096>\n"
                        "Examples:\n"
                        "      $> ./lab1.1-matrixmul 128\n"
                        );
        exit(1);
    }

    // Note: Matrix width and height must be multiples of block size.
    if (!strcmp(matrix_id, "8")) {
        Mw = Mh = Nw = Nh = Pw = Ph = 8;
        block_size = 2; // thread number = block_size^2
        input_fn = strdup("matrix_8.bin");
        gold_fn = strdup("matrix_8.gold");
        if_quiet = false; // If not display matrix contents
    } else
    if (!strcmp(matrix_id, "128")) {
        Mw = Mh = Nw = Nh = Pw = Ph = 128;
        block_size = 16; // thread number = block_size^2
        input_fn = strdup("matrix_128.bin");
        gold_fn = strdup("matrix_128.gold");
        if_quiet = true; // If not display matrix contents
    } else
    if (!strcmp(matrix_id, "512")) {
        Mw = Mh = Nw = Nh = Pw = Ph = 512;
        block_size = 16; // thread number = block_size^2
        input_fn = strdup("matrix_512.bin");
        gold_fn = strdup("matrix_512.gold");
        if_quiet = true; // If not display matrix contents
    } else
    if (!strcmp(matrix_id, "3072")) {
        Mw = Mh = Nw = Nh = Pw = Ph = 3072;
        block_size = 16; // thread number = block_size^2
        input_fn = strdup("matrix_3072.bin");
        gold_fn = strdup("matrix_3072.gold");
        if_quiet = true; // If not display matrix contents
    } else
    if (!strcmp(matrix_id, "4096")) {
        Mw = Mh = Nw = Nh = Pw = Ph = 4096;
        block_size = 16; // thread number = block_size^2
        input_fn = strdup("matrix_4096.bin");
        gold_fn = strdup("matrix_4096.gold");
        if_quiet = true; // If not display matrix contents
    } else {
        printf("***Error on %s: %d: Undefined matrix ID.\n",
            __FILE__, __LINE__);
        printf("   You should add it to the source code.\n");
        printf("   Current available ID's are 8, 128, 512, 3072, 4096.\n");
        exit(1);
    }

    printf("Input matrix file name: %s\n", input_fn);
    printf("Setup host side environment and launch kernel:\n");

    // allocate host memory for matrices M and N
    printf("  Allocate host memory for matrices M and N.\n");
    printf("    M: %d x %d\n", Mw, Mh);
    printf("    N: %d x %d\n", Nw, Nh);
    unsigned int size_M = Mw * Mh;
    unsigned int mem_size_M = sizeof(float) * size_M;
    float* hostM = (float*) malloc(mem_size_M);
    unsigned int size_N = Nw * (Nh);
    unsigned int mem_size_N = sizeof(float) * size_N;
    float* hostN = (float*) malloc(mem_size_N);

    // allocate memory for the result on host side
    printf("  Allocate memory for the result on host side.\n");
    unsigned int size_P = Pw * Ph;
    unsigned int mem_size_P = sizeof(float) * size_P;
    float* hostP = (float*) malloc(mem_size_P);

    // Initialize the input matrices.
    printf("  Initialize the input matrices.\n");
    unsigned int * matrix = ReadMatrixFile(input_fn, Pw, Ph, if_quiet);
    for (i = 0; i < Mw; i++)
        for (j = 0; j < Nw; j++)
	        hostM[i * Mw + j] = hostN[i * Mw + j] = (float) matrix[i*Mw + j];
    free(matrix); matrix = NULL;

    // ===================================================================
    //  Allocate device memory for the input matrices.
    //  Copy memory from the host memory to the device memory.
    // ===================================================================

    // Start measuring transfer times from CPU to GPU
    cudaEventRecord(start, NULL);

    printf("  Allocate device memory.\n");
    cudaMalloc((void**) &deviceM, mem_size_M );
    cudaMalloc((void**) &deviceN, mem_size_N );

    printf("  Copy host memory data to device.\n");
    cudaMemcpy( deviceM, hostM, mem_size_M, cudaMemcpyHostToDevice);
    cudaMemcpy( deviceN, hostN, mem_size_N, cudaMemcpyHostToDevice);

    printf("  Allocate device memory for results and clean it.\n");
    cudaMalloc((void**) &deviceP, mem_size_P);
    cudaMemset(deviceP, 0, mem_size_P);

    // Stop measuring transfer times from CPU to GPU
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timer_memory_in, start, stop);

    // ===================================================================
    // Initialize the thread block and kernel grid dimensions
    // and invoke the CUDA kernel.
    // You may assume that each matrix dimension is a multiple
    // of the defined constant block_size.
    // ===================================================================

    printf("  Setup kernel execution parameters.\n");

    // Different ways of declarations
    #if 1
    dim3 block;
    dim3 grid;
    grid.x = Pw/block_size;
    grid.y = Pw/block_size;
    block.x = block_size;
    block.y = block_size;
    #else
    dim3 block(block_size, block_size);
    dim3 grid(Pw/block.x, Pw/block.y);
    #endif

    printf("  # of threads in a block: %d x %d (%d)\n",
        block.x, block.y, block.x * block.y);
    printf("  # of blocks in a grid  : %d x %d (%d)\n",
        grid.x, grid.y, grid.x * grid.y);

    printf("  Executing the kernel...\n");
    // Start measuring the computation time for the CUDA kernel
    cudaEventRecord(start, NULL);

    // Invoke the CUDA kernel here
    matrixMul<<<grid, block>>> (deviceP, deviceM, deviceN, Mh, Mw, Nw, block_size);

    // Make sure all threads have finished their jobs
    // before we stop the timer_compute.
    cudaThreadSynchronize();

    // Stop measuring the computation time for the CUDA kernel
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timer_compute, start, stop);

    // ===================================================================

    // ===================================================================
    // Copy the results back from the host
    // ===================================================================

    printf("  Copy result from device to host.\n");
    // Start measuring the transfer time back from the GPU to the CPU
    cudaEventRecord(start, NULL);
    cudaMemcpy( hostP, deviceP, mem_size_P, cudaMemcpyDeviceToHost);
    // Stop measuring the transfer time back from the GPU to the CPU
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timer_memory_out, start, stop);

    // ================================================
    // Show timing information
    // ================================================

    printf("Transfer time from CPU to GPU:  %.4f ms.\n", timer_memory_in);
    printf("GPU computation time:           %.4f ms.\n", timer_compute);
    printf("Transfer time from GPU to CPU:  %.4f ms.\n\n", timer_memory_out);
    
    printf("Total GPU processing time:      %.4f ms.\n", timer_memory_in+timer_compute+timer_memory_out);

    // ===========================
    // Compare CPU and GPU results 
    // ===========================

    // Full result check when input matrix is <= 512x512
    //if (0) {
    if (Mw * Nw > 512*512) {
        printf("\nInput matrix size is too big. Skip computing reference.\n");

    } else {
        printf("\nCheck results with those computed by CPU.\n");
        printf ("  Computing reference solution.\n");

        // Start measuring the computation time for the CPU
        cudaEventRecord(start, NULL);

        float* reference = (float*) malloc(mem_size_P);
        computeGold(reference, hostM, hostN, Mh, Mw, Nw);

        // Stop measuring the computation time for the CPU
        cudaEventRecord(stop, NULL);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&timer_compute, start, stop);

        printf("  CPU Processing time   : %.4f ms.\n\n", timer_compute);

        printf("  CPU checksum: %g\n", CheckSum(reference, Mw, Nw));

        matrix = (unsigned int *) malloc(Pw * Ph * sizeof(unsigned int));
        for (i = 0; i < Ph; i++)
            for (j = 0; j < Pw; j++)
                matrix[i*Pw + j] = (unsigned int) reference[i*Pw + j];

        WriteMatrixFile("lab1.1-matrixmul.gold", matrix, Pw, Ph, 1);
        free(matrix); matrix = NULL;
        free(reference);
    }

    printf("  GPU checksum: %g\n", CheckSum(hostP, Mw, Nw));

    /* Write matrix C to output binary file */
    matrix = (unsigned int *) malloc (Pw * Ph * sizeof(unsigned int));
    for (i = 0; i < Ph; i++)
        for (j = 0; j < Pw; j++)
            matrix[i*Pw + j] = (unsigned int) hostP[i*Pw + j];
    WriteMatrixFile("lab1.1-matrixmul.bin", matrix, Pw, Ph, 1);
    free (matrix); matrix = NULL;

    if (Mw >= 3072 && Mh >= 3072) {
        CompareMatrixFile("lab1.1-matrixmul.bin", gold_fn, Pw, Ph, if_quiet);
    } else {
        CompareMatrixFile("lab1.1-matrixmul.bin", "lab1.1-matrixmul.gold",
            Pw, Ph, if_quiet);
    }

    // clean up memory
    free(hostM); free(hostN); free(hostP);
    free(input_fn); free(gold_fn);

    // ===================================================================
    // Free the device memory
    // ===================================================================
    cudaFree( deviceM );
    cudaFree( deviceN );
    cudaFree( deviceP );
}


