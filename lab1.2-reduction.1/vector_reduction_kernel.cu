#define NUM_ELEMENTS 512


// CUDA kernel to perform the reduction in parallel on the GPU
//! @param g_idata  input data in global memory
//                  result is expected in index 0 of g_idata
//! @param n        input number of elements to scan from input data
__global__ void reduction(float *g_data, int n)
{
  int stride;
  // Define shared memory
  __shared__ float scratch[NUM_ELEMENTS];

  // Load the shared memory
  int gindex = blockIdx.x * blockDim.x + threadIdx.x;
  int lindex = threadIdx.x;

  scratch[ lindex ] = g_data[ gindex ];
  if(threadIdx.x + blockDim.x < n)
    scratch[ lindex + n/2 ] = g_data[ gindex + n/2];
  
  __syncthreads();

  int selectorEsquema = 3;

  if (selectorEsquema == 1) { // ESQUEMA 1
    // Do sum reduction on the shared memory data
    for( stride=NUM_ELEMENTS/2; stride>=1; stride = stride/2 ) /* COMPLETAR 2 (el bucle de la reducci�n): Se ha dejado como ejemplo el caso correspondiente al esquema de reducci�n 1. Hay que cambiarlo si escogemos el esquema 2 o el 3 */
    {
      /* COMPLETAR 2 (el cuerpo de la reducci�n en los dos renglones siguientes, seg�n hayamos escogido el esquema de reducci�n 1, 2 � 3) */ 
      if ( lindex < stride )
        scratch[ lindex ] += scratch[ lindex + stride ];
      __syncthreads();
    }
  } else if (selectorEsquema == 2) { // ESQUEMA 2
    for (stride = 1; stride < NUM_ELEMENTS; stride *= 2) 
    {
      int i = 2 * stride * lindex;
      
      if ( i < NUM_ELEMENTS )
        scratch[ i ] += scratch[ i + stride ];
      __syncthreads();
    }

  } else if (selectorEsquema == 3) { // ESQUEMA 3
    for( stride=NUM_ELEMENTS; stride>=1; stride = stride/2 )
    {
      if ( lindex < stride / 2 )
        scratch[ lindex ] += scratch[ stride - lindex - 1 ];
      __syncthreads();
    }
  }
  

 // Store results back to global memory
  if(threadIdx.x == 0)
    g_data[0] = scratch[0];

  return;
}
