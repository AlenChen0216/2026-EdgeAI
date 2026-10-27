#pragma once
// =============================================================================
//  custom.cuh  --  Your own optimized kernels (Part 4)
// =============================================================================
//  Row-major, FP32:  C (MxN) = alpha * A (MxK) @ B (KxN) + beta * C
//  YOU EDIT THIS FILE.
// =============================================================================
#include "../common.cuh"


template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
    sgemm_custom_vec4(int M, int N, int K, float alpha, const float* __restrict__ A,
                const float* __restrict__ B, float beta,
                float* __restrict__ C) {
  constexpr int NT = (BM / TM) * (BN / TN);  // threads per block
  // As is stored TRANSPOSED: As[k][m] holds A[m][k] of the current tile.
  __shared__ __align__(16) float As[BK][BM];
  __shared__ __align__(16) float Bs[BK][BN];
  const int tid = threadIdx.x;
  // This thread computes rows tr*TM .. tr*TM+TM-1 and
  // columns tc*TN .. tc*TN+TN-1 of the block's BM x BN tile.
  const int tr = tid / (BN / TN), tc = tid % (BN / TN);
  float acc[TM][TN] = {};
  float ra[TM], rb[TN];
  for (int k0 = 0; k0 < K; k0 += BK) {
    // Each thread loads four adjacent FP32 values in one aligned access.
    // A is scattered into the existing transposed shared-memory layout.
    for (int v = tid; v < BM * (BK / 4); v += NT) {
      const int m = v / (BK / 4);
      const int k = (v % (BK / 4)) * 4;
      const size_t row = (size_t)blockIdx.y * BM + m;
      const float4 a = *reinterpret_cast<const float4*>(A + row * K + k0 + k);
      As[k + 0][m] = a.x;
      As[k + 1][m] = a.y;
      As[k + 2][m] = a.z;
      As[k + 3][m] = a.w;
    }
    for (int v = tid; v < BK * (BN / 4); v += NT) {
      const int k = v / (BN / 4);
      const int n = (v % (BN / 4)) * 4;
      const size_t row = (size_t)k0 + k;
      const size_t col = (size_t)blockIdx.x * BN + n;
      *reinterpret_cast<float4*>(&Bs[k][n]) =
          *reinterpret_cast<const float4*>(B + row * N + col);
    }
    __syncthreads();
    // GIVEN: outer product. Per k, each thread loads TM values of A and TN
    // values of B into registers and does TM*TN FMAs with them.
    for (int k = 0; k < BK; ++k) {
      for (int i = 0; i < TM; ++i) ra[i] = As[k][tr * TM + i];
      for (int j = 0; j < TN; ++j) rb[j] = Bs[k][tc * TN + j];
      for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j) acc[i][j] += ra[i] * rb[j];
    }
    __syncthreads();
  }
  // Keep scalar output stores to isolate the input-load optimization.
  for(int i = 0; i < TM; ++i){
    for(int j = 0; j < TN; ++j){
      const int row = blockIdx.y * BM + tr * TM + i;
      const int col = blockIdx.x * BN + tc * TN + j;
      const size_t idx = (size_t)row * N + col;
      C[idx] = alpha * acc[i][j] + beta * C[idx];
    }
  }
}

// e.g. a vectorized / double-buffered / swizzled variant of the reg2d kernel,
// a shape-dispatching launcher, ... Register each one in src/kernels.cuh.
template <int BM, int BN, int BK, int TM, int TN>
bool launch_custom_vec4(int M, int N, int K, float alpha, const float* A,
                  const float* B, float beta, float* C, cudaStream_t s) {
    
  static_assert(BM % TM == 0 && BN % TN == 0, "tile must divide block tile");
  constexpr int kThreads = (BM / TM) * (BN / TN);
  static_assert(kThreads <= 1024, "too many threads per block");
  static_assert(BK % 4 == 0 && BN % 4 == 0, "float4 requires four-wide tiles");
  if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)) % 16)
    return false;
  if (M % BM || N % BN || K % BK) return false;
  dim3 grid(N / BN, M / BM);
  sgemm_custom_vec4<BM, BN, BK, TM, TN> <<<grid, kThreads, 0, s>>>(M, N, K, alpha, A, B, beta, C);
    return true;

}


// Warp tiling on top of custom_vec4 (no shared-memory swizzling).
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
    sgemm_custom_vec4_warptiling(int M, int N, int K, float alpha, const float* __restrict__ A,
                const float* __restrict__ B, float beta,
                float* __restrict__ C) {
  constexpr int NT = (BM / TM) * (BN / TN);  // threads per block
  // As is stored TRANSPOSED: As[k][m] holds A[m][k] of the current tile.
  __shared__ __align__(16) float As[BK][BM];
  __shared__ __align__(16) float Bs[BK][BN];
  const int tid = threadIdx.x;
  // 32x64 outputs per warp, with four 4x4 sub-tiles per lane.
  // lane = [4 rows x 8 columns]; each sub-tile is 16x32 across the warp.
  static_assert(TM == 8 && TN == 8 && BM % 32 == 0 && BN % 64 == 0,
                "warp layout requires 8x8 thread tiles and 32x64 warp tiles");
  constexpr int WM = 32, WN = 64;
  const int warp = tid / 32, lane = tid % 32;
  const int warpRow = (warp / (BN / WN)) * WM;
  const int warpCol = (warp % (BN / WN)) * WN;
  const int laneRow = lane / 8, laneCol = lane % 8;
  float acc[TM][TN] = {};
  float ra[TM], rb[TN];
  for (int k0 = 0; k0 < K; k0 += BK) {
    // Each thread loads four adjacent FP32 values in one aligned access.
    // A is scattered into the existing transposed shared-memory layout.
    for (int v = tid; v < BM * (BK / 4); v += NT) {
      const int m = v / (BK / 4);
      const int k = (v % (BK / 4)) * 4;
      const size_t row = (size_t)blockIdx.y * BM + m;
      const float4 a = *reinterpret_cast<const float4*>(A + row * K + k0 + k);
      As[k + 0][m] = a.x;
      As[k + 1][m] = a.y;
      As[k + 2][m] = a.z;
      As[k + 3][m] = a.w;
    }
    for (int v = tid; v < BK * (BN / 4); v += NT) {
      const int k = v / (BN / 4);
      const int n = (v % (BN / 4)) * 4;
      const size_t row = (size_t)k0 + k;
      const size_t col = (size_t)blockIdx.x * BN + n;
      *reinterpret_cast<float4*>(&Bs[k][n]) =
          *reinterpret_cast<const float4*>(B + row * N + col);
    }
    __syncthreads();
    // GIVEN: outer product. Per k, each thread loads TM values of A and TN
    // values of B into registers and does TM*TN FMAs with them.
    for (int k = 0; k < BK; ++k) {
      for (int i = 0; i < TM; ++i) ra[i] = As[k][warpRow + (i / 4) * 16 + laneRow * 4 + i % 4];
      for (int j = 0; j < TN; ++j) rb[j] = Bs[k][warpCol + (j / 4) * 32 + laneCol * 4 + j % 4];
      for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j) acc[i][j] += ra[i] * rb[j];
    }
    __syncthreads();
  }
  // Keep scalar output stores to isolate the input-load optimization.
  for(int i = 0; i < TM; ++i){
    for(int j = 0; j < TN; ++j){
      const int row = blockIdx.y * BM + warpRow + (i / 4) * 16 + laneRow * 4 + i % 4;
      const int col = blockIdx.x * BN + warpCol + (j / 4) * 32 + laneCol * 4 + j % 4;
      const size_t idx = (size_t)row * N + col;
      C[idx] = alpha * acc[i][j] + beta * C[idx];
    }
  }
}

// Same tile loads and launch geometry as custom_vec4; only output ownership changes.
template <int BM, int BN, int BK, int TM, int TN>
bool launch_custom_vec4_warptiling(int M, int N, int K, float alpha, const float* A,
                  const float* B, float beta, float* C, cudaStream_t s) {
    
  static_assert(BM % TM == 0 && BN % TN == 0, "tile must divide block tile");
  constexpr int kThreads = (BM / TM) * (BN / TN);
  static_assert(kThreads <= 1024, "too many threads per block");
  static_assert(BK % 4 == 0 && BN % 4 == 0, "float4 requires four-wide tiles");
  if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)) % 16)
    return false;
  if (M % BM || N % BN || K % BK) return false;
  dim3 grid(N / BN, M / BM);
  sgemm_custom_vec4_warptiling<BM, BN, BK, TM, TN> <<<grid, kThreads, 0, s>>>(M, N, K, alpha, A, B, beta, C);
    return true;

}
