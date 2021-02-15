#ifndef _MATRIXMUL_H_
#define _MATRIXMUL_H_

// ==================================================================
// These are the five performance-tuning parameters at your disposal

// Thread block size
#define BLOCK_SIZE 32

// Dot product loop unrolling factor
#define UNROLL 16  // Available values are 0, 2, 4, and 16.

//Register spilling (define == On, undef == Off)
#undef SPILL
//#define SPILL

//Prefetching (define == On, undef == Off)
//#undef PREFETCH
#define PREFETCH

// End performance-tuning parameters
// ==================================================================

extern "C"
void computeGold(float* P, const float* M, const float* N, int Mh, int Mw, int Nw);

#endif // _MATRIXMUL_H_

