
#include <cublas_v2.h>

// clang-format off
#include "tensors/gpu/prod.h"
#include "tensors/gpu/backend.h"
#include "tensors/gpu/cuda_helpers.h"
// clang-format on

namespace marian {

namespace gpu {

void Prod(marian::Tensor C,
          const marian::Tensor A,
          const marian::Tensor B,
          bool transA,
          bool transB,
          float beta,
          float scalar) {
  cudaSetDevice(C->getDevice().no);
  float alpha = scalar;

  size_t m = A->shape().elements() / A->shape().back();
  size_t k = A->shape().back();
  if(transA)
    std::swap(m, k);

  size_t l = B->shape().elements() / B->shape().back();
  size_t n = B->shape().back();
  if(transB)
    std::swap(l, n);

  size_t lda = A->shape().back();
  size_t ldb = B->shape().back();
  size_t ldc = B->shape().back();

  if(transB)
    ldc = B->shape().elements() / B->shape().back();

  cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

  auto cublasHandle = std::static_pointer_cast<gpu::Backend>(C->getBackend())
                          ->getCublasHandle();

#if CUDA_VERSION >= 9000
// cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH);
#endif

  cublasSgemm(cublasHandle,
              opB,
              opA,
              n,
              m,
              k,
              &alpha,
              B->data(),
              ldb,
              A->data(),
              lda,
              &beta,
              C->data(),
              ldc);
#if CUDA_VERSION >= 9000
// cublasSetMathMode(cublasHandle, CUBLAS_DEFAULT_MATH);
#endif
}

void ProdWithBias(marian::Tensor C,
          const marian::Tensor A,
          const marian::Tensor B,
          const marian::Tensor bias,
          bool transA,
          bool transB,
          float beta,
          float scalar) {
  marian::gpu::Prod(C, A, B, transA, transB, beta, scalar);
  marian::gpu::Add(functional::_1, 1.f, C, bias);
}


void ProdBatched(marian::Tensor C,
                 const marian::Tensor A,
                 const marian::Tensor B,
                 bool transA,
                 bool transB,
                 float beta,
                 float scalar) {
  cudaSetDevice(C->getDevice().no);
  float alpha = scalar;

  size_t batchA = A->shape().elements() / (A->shape()[-1] * A->shape()[-2]);
  size_t batchB = B->shape().elements() / (B->shape()[-1] * B->shape()[-2]);

  size_t m = A->shape()[-2];
  size_t k = A->shape()[-1];
  if(transA)
    std::swap(m, k);

  size_t l = B->shape()[-2];
  size_t n = B->shape()[-1];
  if(transB)
    std::swap(l, n);

  size_t lda = A->shape()[-1];
  size_t ldb = B->shape()[-1];
  size_t ldc = B->shape()[-1];

  if(transB)
    ldc = B->shape()[-2];

  cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

  auto cublasHandle = std::static_pointer_cast<gpu::Backend>(C->getBackend())
                          ->getCublasHandle();

#if CUDA_VERSION >= 9000
// cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH);
#endif
  cublasSgemmStridedBatched(cublasHandle,
                            opB,
                            opA,
                            n,
                            m,
                            k,
                            &alpha,
                            B->data(),
                            ldb,
                            batchB == 1 ? 0 : n * k,
                            A->data(),
                            lda,
                            batchA == 1 ? 0 : m * k,
                            &beta,
                            C->data(),
                            ldc,
                            n * m,
                            std::max(batchA, batchB));
#if CUDA_VERSION >= 9000
// cublasSetMathMode(cublasHandle, CUBLAS_DEFAULT_MATH);
#endif
}

void ProdBatched(marian::Tensor C,
                 const marian::Tensor A,
                 const marian::Tensor B,
                 const std::vector<size_t>& indicesA,
                 const std::vector<size_t>& indicesB,
                 bool transA,
                 bool transB,
                 float beta,
                 float scalar) {
  cudaSetDevice(C->getDevice().no);
  float alpha = scalar;

  size_t batchA = A->shape().elements() / (A->shape()[-1] * A->shape()[-2]);
  size_t batchB = B->shape().elements() / (B->shape()[-1] * B->shape()[-2]);
  const size_t batchCount = std::max(batchA, batchB);

  size_t m = A->shape()[-2];
  size_t k = A->shape()[-1];
  if(transA)
    std::swap(m, k);

  size_t l = B->shape()[-2];
  size_t n = B->shape()[-1];
  if(transB)
    std::swap(l, n);

  size_t lda = A->shape()[-1];
  size_t ldb = B->shape()[-1];
  size_t ldc = B->shape()[-1];

  if(transB)
    ldc = B->shape()[-2];

  float **a_ptrs_host, **b_ptrs_host, **c_ptrs_host;
  a_ptrs_host = new float*[batchA];
  b_ptrs_host = new float*[batchB];
  c_ptrs_host = new float*[batchCount];
  for(int i = 0; i < std::max(batchA, batchB); ++i) {
    a_ptrs_host[i] = A->data() + (indicesA.size() > 0 ? indicesA[i] * m * k : 0);
    b_ptrs_host[i] = B->data() + (indicesB.size() > 0 ? indicesB[i] * l * n : 0);
    // c_ptrs_host[i] = C->data() + i * n * m;
    c_ptrs_host[i] = (i > 3 ? C->data() + i * n * m : C->data());
  }

  float **a_ptrs, **b_ptrs, **c_ptrs;
  CUDA_CHECK(cudaMalloc((void**)&a_ptrs, batchA * sizeof(float*)));
  CUDA_CHECK(cudaMalloc((void**)&b_ptrs, batchB * sizeof(float*)));
  CUDA_CHECK(cudaMalloc((void**)&c_ptrs, batchCount * sizeof(float*)));
  cudaMemcpy(a_ptrs, a_ptrs_host, batchA * sizeof(float*), cudaMemcpyHostToDevice);
  cudaMemcpy(b_ptrs, b_ptrs_host, batchB * sizeof(float*), cudaMemcpyHostToDevice);
  cudaMemcpy(c_ptrs, c_ptrs_host, batchCount * sizeof(float*), cudaMemcpyHostToDevice);

  cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

  auto cublasHandle = std::static_pointer_cast<gpu::Backend>(C->getBackend())
                          ->getCublasHandle();

#if CUDA_VERSION >= 9000
// cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH);
#endif
  cublasSgemmBatched(cublasHandle,
                     opB,
                     opA,
                     n,
                     m,
                     k,
                     &alpha,
                     (const float**)b_ptrs, // TODO Why do I have to explicitly cast to const?
                     ldb,
                     (const float**)a_ptrs, // TODO - compiler warning otherwise
                     lda,
                     &beta,
                     c_ptrs, // NOTE
                     ldc,
                     batchCount);
#if CUDA_VERSION >= 9000
// cublasSetMathMode(cublasHandle, CUBLAS_DEFAULT_MATH);
#endif

  CUDA_CHECK(cudaFree(a_ptrs));
  CUDA_CHECK(cudaFree(b_ptrs));
  CUDA_CHECK(cudaFree(c_ptrs));
}
}
}
