#ifndef _MATRIXMUL_H_
#define _MATRIXMUL_H_

// Thread block size
#define BLOCK_SIZE  16  // 4, 8, 16

extern "C"
void computeGold(float* P, const float* M, const float* N, int Mh, int Mw, int Nw);

#endif // _MATRIXMUL_H_

