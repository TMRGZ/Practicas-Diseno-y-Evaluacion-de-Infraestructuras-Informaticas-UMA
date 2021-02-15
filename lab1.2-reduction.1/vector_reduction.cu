// includes, system
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

// includes, kernels
#include "vector_reduction_kernel.cu"

// For simplicity, just to get the idea in this MP, we're fixing the problem size to 512 elements.
#define NUM_ELEMENTS 512*1

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

float computeOnDevice(float* h_data, int array_mem_size);

extern "C" 
void computeGold( float* reference, float* idata, const unsigned int len);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{

cudaSetDevice(0);
    runTest( argc, argv);
    return EXIT_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
//! Run naive scan test
////////////////////////////////////////////////////////////////////////////////
void runTest( int argc, char** argv) 
{
    int num_elements = NUM_ELEMENTS;

    const unsigned int array_mem_size = sizeof( float) * num_elements;

    // Allocate host memory to store the input data
    float* h_data = (float*) malloc( array_mem_size);

    // initialize the input data on the host to be integer values
    // between 0 and 1000
    for( unsigned int i = 0; i < num_elements; ++i) 
        h_data[i] = floorf(1000*(rand()/(float)RAND_MAX));

    // Function to compute the reference solution on CPU using a C sequential version of the algorithm
    // It is written in the file "vector_reduction_gold.cpp". The Makefile compiles this file too.
    float reference = 0.0f;  
    computeGold(&reference , h_data, num_elements);
    
    // Function to compute the solution on GPU using a call to a CUDA kernel (see body below)
    // The kernel is written in the file "vector_reduction_kernel.cu". The Makefile also compiles this file.
    float result = computeOnDevice(h_data, num_elements);

    // We can use an epsilon of 0 since values are integral and in a range that can be exactly represented
    float epsilon = 0.0f;
    unsigned int result_regtest = (abs(result - reference) <= epsilon);
    printf( "Test %s\n", (1 == result_regtest) ? "CORRECTO: Coinciden los resultados de la CPU y la GPU" : "INCORRECTO: Los resultados calculados en paralelo en la GPU no coinciden con los obtenidos secuencialmente en la CPU");
    printf( "device: %f  host: %f\n", result, reference);
    // cleanup memory
    free( h_data);
}

// Function to call the CUDA kernel on the GPU.
// Take h_data from host, copies it to device, setup grid and thread 
// dimensions, excutes kernel function, and copy result of scan back
// to h_data.
// Note: float* h_data is both the input and the output of this function.
float computeOnDevice(float* h_data, int num_elements)
{
  float* d_data = NULL;
  float result;

  // Memory allocation on device side
  cudaMalloc((void**)&d_data, num_elements );

  // Copy from host memory to device memory
  cudaMemcpy(d_data, h_data, num_elements, cudaMemcpyHostToDevice);
 
  int threads = (num_elements/2) + num_elements%2;

  // Invoke the kernel
  reduction<<<1,threads>>>(d_data,num_elements);

  // Copy from device memory back to host memory
  cudaMemcpy(&result, d_data, sizeof(float), cudaMemcpyDeviceToHost );
  cudaFree(d_data);
  cudaDeviceReset();
  return result;
}
     

