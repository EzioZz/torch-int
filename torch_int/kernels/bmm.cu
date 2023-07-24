#include "include/bmm.h"
#include "include/common.h"
#include <cutlass/core_io.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/host_tensor.h>
#include <cublas_v2.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/Exceptions.h>

torch::Tensor bmm_s8t_s8n_f32t(torch::Tensor A, torch::Tensor B, float alpha) {
  int batch_size = A.size(0);
  int M = A.size(1);
  int N = B.size(1);
  int K = A.size(2);

  auto C = torch::empty({batch_size, M, N},
                        torch::dtype(torch::kFloat32).device(A.device()));
  int lda = A.size(2);
  int ldb = B.size(2);
  int ldc = C.size(2);

  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::RowMajor;

  using ElementOutput = float;
  using ElementInputA = int8_t;
  using ElementInputB = int8_t;
  using ElementAccumulator = int32_t;
  using ElementComputeEpilogue = float;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator, ElementComputeEpilogue>;

  using Gemm = cutlass::gemm::device::GemmBatched<
      ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput,
      LayoutOutput, ElementAccumulator, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<256, 128, 64>,
      cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 32>,
      EpilogueOp>;

  long long int batch_stride_A = M * K;
  long long int batch_stride_B = N * K;
  long long int batch_stride_C = M * N;

  Gemm gemm_op;
  typename Gemm::Arguments arguments{
      {M, N, K},      {A.data_ptr<ElementInputA>(), lda},
      batch_stride_A, {B.data_ptr<ElementInputB>(), ldb},
      batch_stride_B, {C.data_ptr<ElementOutput>(), ldc},
      batch_stride_C, {C.data_ptr<ElementOutput>(), ldc},
      batch_stride_C, {alpha, 0},
      batch_size};

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check the problem size is supported or not
  cutlass::Status status = gemm_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot implement");
  }

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot initialize");
  }

  status = gemm_op();
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot run");
  }
  return C;
}

torch::Tensor bmm_s8t_s8n_s8t(torch::Tensor A, torch::Tensor B, float alpha) {
  int batch_size = A.size(0);
  int M = A.size(1);
  int N = B.size(1);
  int K = A.size(2);

  auto C = torch::empty({batch_size, M, N},
                        torch::dtype(torch::kInt8).device(A.device()));
  int lda = A.size(2);
  int ldb = B.size(2);
  int ldc = C.size(2);

  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::RowMajor;

  using ElementOutput = int8_t;
  using ElementInputA = int8_t;
  using ElementInputB = int8_t;
  using ElementAccumulator = int32_t;
  using ElementComputeEpilogue = float;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombinationClamp<
      ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator, ElementComputeEpilogue>;

  using Gemm = cutlass::gemm::device::GemmBatched<
      ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput,
      LayoutOutput, ElementAccumulator, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<256, 128, 64>,
      cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 32>,
      EpilogueOp>;

  long long int batch_stride_A = M * K;
  long long int batch_stride_B = N * K;
  long long int batch_stride_C = M * N;

  Gemm gemm_op;
  typename Gemm::Arguments arguments{
      {M, N, K},      {A.data_ptr<ElementInputA>(), lda},
      batch_stride_A, {B.data_ptr<ElementInputB>(), ldb},
      batch_stride_B, {C.data_ptr<ElementOutput>(), ldc},
      batch_stride_C, {C.data_ptr<ElementOutput>(), ldc},
      batch_stride_C, {alpha, 0},
      batch_size};

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check the problem size is supported or not
  cutlass::Status status = gemm_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot implement");
  }

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot initialize");
  }

  status = gemm_op();
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot run");
  }
  return C;
}

torch::Tensor bmm_s8t_s8n_s32t(torch::Tensor A, torch::Tensor B) {
  int batch_size = A.size(0);
  int M = A.size(1);
  int N = B.size(1);
  int K = A.size(2);

  auto C = torch::empty({batch_size, M, N},
                        torch::dtype(torch::kInt32).device(A.device()));
  int lda = A.size(2);
  int ldb = B.size(2);
  int ldc = C.size(2);

  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::RowMajor;

  using ElementOutput = int32_t;
  using ElementInputA = int8_t;
  using ElementInputB = int8_t;
  using ElementAccumulator = int32_t;
  using ElementComputeEpilogue = int32_t;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator, ElementComputeEpilogue>;

  using Gemm = cutlass::gemm::device::GemmBatched<
      ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput,
      LayoutOutput, ElementAccumulator, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<256, 128, 64>,
      cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 32>,
      EpilogueOp>;

  long long int batch_stride_A = M * K;
  long long int batch_stride_B = N * K;
  long long int batch_stride_C = M * N;

  Gemm gemm_op;

  ElementComputeEpilogue alpha = 1;

  cutlass::Status status = gemm_op({{M, N, K},
                                    {A.data_ptr<ElementInputA>(), lda},
                                    batch_stride_A,
                                    {B.data_ptr<ElementInputB>(), ldb},
                                    batch_stride_B,
                                    {C.data_ptr<ElementOutput>(), ldc},
                                    batch_stride_C,
                                    {C.data_ptr<ElementOutput>(), ldc},
                                    batch_stride_C,
                                    {alpha, 0},
                                    batch_size});

  if (status != cutlass::Status::kSuccess) {
    std::cout << "cutlass error code: " << (int)status << std::endl;
  }
  return C;
}

torch::Tensor bmm_s8t_s8n_s32t_cublas(torch::Tensor A, torch::Tensor B) {
  int batch_size = A.size(0);
  int M = A.size(1);
  int N = B.size(1); // b, N, K
  int K = A.size(2); 
  auto C = torch::empty({batch_size, M, N},
                        torch::dtype(torch::kInt32).device(A.device()));

  int lda = A.size(2); // K
  int ldb = B.size(2); // N
  int ldc = C.size(2); // N

  // using LayoutInputA = cutlass::layout::RowMajor;
  // using LayoutInputB = cutlass::layout::ColumnMajor;
  // using LayoutOutput = cutlass::layout::RowMajor;

  // using ElementOutput = int32_t;
  // using ElementInputA = int8_t;
  // using ElementInputB = int8_t;

  long long int batch_stride_A = M * K;
  long long int batch_stride_B = N * K;
  long long int batch_stride_C = M * N;

  cublasStatus_t status;

  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();


  int32_t alpha = 1; 
  int32_t beta = 0; 

  status = cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                    N, M, K,
                                      (const void*)&alpha, 
                                      (const void*)B.data_ptr(), CUDA_R_8I, K, batch_stride_B,
                                      (const void*)A.data_ptr(), CUDA_R_8I, K, batch_stride_A,
                                      (const void*)&beta,
                                      (void*)C.data_ptr(), CUDA_R_32I, N, batch_stride_C,
                                      batch_size,
                                      CUDA_R_32I, 
                                      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  if (status != CUBLAS_STATUS_SUCCESS) {
      printf("cuBLAS API failed with status %d\n", status);
  }


  return C;
  // cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t handle,
  //                             cublasOperation_t transa,
  //                             cublasOperation_t transb,
  //                             int m,
  //                             int n,
  //                             int k,
  //                             const void    *alpha,
  //                             const void     *A,
  //                             cudaDataType Atype,
  //                             int lda,
  //                             long long int strideA,
  //                             const void     *B,
  //                             cudaDataType Btype,
  //                             int ldb,
  //                             long long int strideB,
  //                             const void    *beta,
  //                             void           *C,
  //                             cudaDataType Ctype,
  //                             int ldc,
  //                             long long int strideC,
  //                             int batchCount,
  //                             cudaDataType computeType,
  //                             cublasGemmAlgo_t algo)

}


torch::Tensor bmm_s8t_s8n_s32t_cublas1batch(torch::Tensor A, torch::Tensor B) {
  int batch_size = A.size(0);
  int M = A.size(1);
  int N = B.size(1); // b, N, K
  int K = A.size(2); 
  auto C = torch::empty({batch_size, M, N},
                        torch::dtype(torch::kInt32).device(A.device()));

  int lda = A.size(2); // K
  int ldb = B.size(2); // N
  int ldc = C.size(2); // N

  // using LayoutInputA = cutlass::layout::RowMajor;
  // using LayoutInputB = cutlass::layout::ColumnMajor;
  // using LayoutOutput = cutlass::layout::RowMajor;

  // using ElementOutput = int32_t;
  // using ElementInputA = int8_t;
  // using ElementInputB = int8_t;

  long long int batch_stride_A = M * K;
  long long int batch_stride_B = N * K;
  long long int batch_stride_C = M * N;

  cublasStatus_t status;

  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();


  int32_t alpha = 1; 
  int32_t beta = 0; 

  status = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                    N, M, K,
                                      (const void*)&alpha, 
                                      (const void*)B.data_ptr(), CUDA_R_8I, K,
                                      (const void*)A.data_ptr(), CUDA_R_8I, K,
                                      (const void*)&beta,
                                      (void*)C.data_ptr(), CUDA_R_32I, N,
                                      CUDA_R_32I, 
                                      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  if (status != CUBLAS_STATUS_SUCCESS) {
      printf("cuBLAS API failed with status %d\n", status);
  }


  return C;
  // cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t handle,
  //                             cublasOperation_t transa,
  //                             cublasOperation_t transb,
  //                             int m,
  //                             int n,
  //                             int k,
  //                             const void    *alpha,
  //                             const void     *A,
  //                             cudaDataType Atype,
  //                             int lda,
  //                             long long int strideA,
  //                             const void     *B,
  //                             cudaDataType Btype,
  //                             int ldb,
  //                             long long int strideB,
  //                             const void    *beta,
  //                             void           *C,
  //                             cudaDataType Ctype,
  //                             int ldc,
  //                             long long int strideC,
  //                             int batchCount,
  //                             cudaDataType computeType,
  //                             cublasGemmAlgo_t algo)

}