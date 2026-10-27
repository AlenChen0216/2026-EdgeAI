#pragma once
// =============================================================================
//  smem.cuh  --  Kernel 3: shared-memory tiling, one output per thread (Part 3a)
// =============================================================================
//  Row-major, FP32:  C (MxN) = alpha * A (MxK) @ B (KxN) + beta * C
//  YOU EDIT THIS FILE.
// =============================================================================
#include "../common.cuh"

// Each block computes a TILE x TILE tile of C. Loop over K in steps of TILE:
// cooperatively load a tile of A and a tile of B into __shared__ memory,
// __syncthreads(), accumulate, __syncthreads().
// Must handle ANY M, N, K (zero-pad out-of-range tile elements).
template <int TILE>
__global__ void sgemm_smem(int M, int N, int K, float alpha,
                           const float* __restrict__ A,
                           const float* __restrict__ B, float beta,
                           float* __restrict__ C) {
  // TODO(Part 3a)
  const int thx = threadIdx.x;
  const int thy = threadIdx.y;
  const int row = blockIdx.x * TILE + thy;
  const int col = blockIdx.y * TILE + thx;
  __shared__ float sA[TILE][TILE];
  __shared__ float sB[TILE][TILE];
  float acc = 0.f;
  for ( int k=0 ;k<K; k+=TILE){
    sA[thy][thx] = (row < M && k + thx < K) ? A[(size_t)row * K + (k + thx)] : 0.f;
    sB[thy][thx] = (col < N && k + thy < K) ? B[(size_t)(k + thy) * N + col] : 0.f;
    __syncthreads();
    for (int i = 0; i < TILE; ++i) {
      acc += sA[thy][i] * sB[i][thx];
    }
    __syncthreads();
  }
  if (row < M && col < N) {
    const size_t idx = (size_t)row * N + col;
    C[idx] = alpha * acc + beta * C[idx];
  }
}

template <int TILE>
bool launch_smem(int M, int N, int K, float alpha, const float* A,
                 const float* B, float beta, float* C, cudaStream_t s) {
  // TODO(Part 3a)
  dim3 block(TILE, TILE);
  dim3 grid(ceil_div(M, TILE), ceil_div(N, TILE));
  sgemm_smem<TILE><<<grid, block, 0, s>>>(M, N, K, alpha, A, B, beta, C);
  return true;
}
