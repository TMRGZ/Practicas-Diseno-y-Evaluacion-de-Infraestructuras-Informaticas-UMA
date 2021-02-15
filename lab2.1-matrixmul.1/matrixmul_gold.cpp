#include <stdio.h>
#include "matrixmul.h"

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! P = M * N
//! @param P          reference data, computed but preallocated
//! @param M          matrix M as provided to device
//! @param N          matrix N as provided to device
//! @param Mh         height of matrix M
//! @param Nw         width of matrix N
////////////////////////////////////////////////////////////////////////////////
void computeGold(float* P, const float* M, const float* N, int Mh, int Mw, int Nw)
{
  int i, j, k;
  float sum, a, b;

  for (i = 0; i < Mh; i++)
    for (j = 0; j < Nw; j++)
      {
	    sum = 0;
	    for (k = 0; k < Mw; k++)
	    {
	        a = M[i * Mw + k];
	        b = N[k * Nw + j];
            //printf ("A[%d] * B[%d]\n", i * Mw + k, k * Nw + j);
	        sum += a * b;
	    }
	    P[i * Nw + j] = (float)sum;
      }
}

