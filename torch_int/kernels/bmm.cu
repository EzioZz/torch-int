#include "include/bmm.h"
#include "include/common.h"
#include <cutlass/core_io.h>
#include <cutlass/cutlass.h>
#include <cutlass/integer_subbyte.h>
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

__global__ void quant_f32_s4(float* a, uint8_t* dst, const size_t sizeA, float scale, float offset) {

    int idx = (threadIdx.x + blockIdx.x * blockDim.x)*4; // 每个线程负责 4 个元素
    float4 reg_a = (reinterpret_cast<float4*>(&a[idx]))[0];
    reg_a.x = (reg_a.x - offset) / scale + offset;
    reg_a.y = (reg_a.x - offset) / scale + offset;
    reg_a.z = (reg_a.z - offset) / scale + offset;
    reg_a.w = (reg_a.w - offset) / scale + offset;

    int4 reg_b;
    reg_b.x = __float2ll_rn(reg_a.x);
    reg_b.y = __float2ll_rn(reg_a.y);
    reg_b.z = __float2ll_rn(reg_a.z);
    reg_b.w = __float2ll_rn(reg_a.w);
    
    int4 reg_c;

    reg_b.x = max(min(127, reg_b.x), -128);
    reg_b.y = max(min(127, reg_b.y), -128);
    reg_b.z = max(min(127, reg_b.z), -128);
    reg_b.w = max(min(127, reg_b.w), -128);

    uint8_t res1 = 0;
    uint8_t res2 = 0;
    res1 = ((reg_b.x & 0x8000) >> 24)|((reg_b.x & 0x0006)<<4) | ((reg_b.y & 0x8000) >> 28)|((reg_b.y & 0x0006)<<4);
    res2 = ((reg_b.z & 0x8000) >> 24)|((reg_b.z & 0x0006)<<4) | ((reg_b.w & 0x8000) >> 28)|((reg_b.w & 0x0006)<<4) ;
    
    // convert 4 float to 4 int4, then store into 2 uint8
    dst[idx>>1 ] = res1;
    dst[idx>>1 + 1] = res2;

    return;
}


// void invokeQuantize(torch::Tensor A, float scale, float offset){
//     // input = torch.clamp(torch.round(input / qinput_interval - offset) + offset, min_v, max_v)
//   int batch_size = A.size(0);
//   int M = A.size(1);
//   int K = A.size(2);
//   size_t sizeA = batch_size * M * K;
  
//   uint8_t* qInt4_A;
//   cudaMalloc((uint8_t**)&qInt4_A, sizeA*sizeof(uint8_t) / 2);

//   constexpr int blockSize = 256;
//   int gridSize = sizeA / blockSize / 4; // 每个线程负责 4 个 float

//   quant_f32_s4<<<gridSize, blockSize>>>(A.data_ptr<float>(), qInt4_A, sizeA, scale, offset);
// }


void benchTestQuantize(torch::Tensor A){
  float scale = 100.0;
  float offset = 20.0;

  int batch_size = A.size(0);
  int M = A.size(1);
  int K = A.size(2);

  size_t sizeA = batch_size * M * K;
  uint8_t* qInt4_A;
  cudaMalloc((uint8_t**)&qInt4_A, sizeA*sizeof(uint8_t) / 2);

  constexpr int blockSize = 256;
  int gridSize = sizeA / blockSize / 4; // 每个线程负责 4 个 float

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  for (int i = 0; i < 100; i++) {
    quant_f32_s4<<<gridSize, blockSize>>>(A.data_ptr<float>(), qInt4_A, sizeA, scale, offset);
  }

  constexpr int n_iter = 1000;
  cudaEventRecord(start);
  for (int i = 0; i < n_iter; i++) {
    quant_f32_s4<<<gridSize, blockSize>>>(A.data_ptr<float>(), qInt4_A, sizeA, scale, offset);
  }
  cudaDeviceSynchronize();
  cudaEventRecord(end);  

  float ms;
  cudaEventElapsedTime(&ms, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);

  long long workload = (long long)n_iter * (sizeA * 4 + sizeA/2 * 1); // byte
  double bandwidth = ((double)workload ) / ((double)ms/1e3) / (1e9);
  printf("Performance: %lfGBS\n", bandwidth);

}


// void benchTestBmmInt4(torch::Tensor A, torch::Tensor B, float alpha){
//   float scale = 100.0;
//   float offset = 20.0;

//   int batch_size = A.size(0);
//   int M = A.size(1);
//   int K = A.size(2);

//   constexpr size_t sizeA = batch_size * M * K;
//   uint8_t* qInt4_A;
//   cudaMalloc((uint8_t**)&qInt4_A, sizeA*sizeof(uint8_t) / 2);

//   constexpr int blockSize = 256;
//   constexpr int gridSize = sizeA / blockSize / 4; // 每个线程负责 4 个 float

//   cudaEvent_t start, end;
//   cudaEventCreate(&start);
//   cudaEventCreate(&end);

//   for (int i = 0; i < 100; i++){
//       float4_add<<<gridSize, blockSize>>>(d_a, d_b, d_c);
//   }

// }

torch::Tensor bmm_s4t_s4n_f32t(torch::Tensor A, torch::Tensor B, float alpha) {
  benchTestQuantize(A);
  // benchTestBmmInt4(A, B, alpha);

  // 注意，这里输入矩阵 A，矩阵 B 都还是 float 类型的
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
  using ElementInputA = cutlass::int4b_t;
  using ElementInputB = cutlass::int4b_t;

  using ElementAccumulator = int32_t;
  using ElementComputeEpilogue = float;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator, ElementComputeEpilogue>;

  using Gemm = cutlass::gemm::device::GemmBatched<
      ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput,
      LayoutOutput, ElementAccumulator, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<256, 128, 64>,
      cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<8, 8, 32>,
      EpilogueOp>;

  long long int batch_stride_A = M * K;
  long long int batch_stride_B = N * K;
  long long int batch_stride_C = M * N;

  Gemm gemm_op;
  typename Gemm::Arguments arguments{
      {M, N, K},      {(const cutlass::int4b_t*)A.data_ptr<int8_t>(), lda},
      batch_stride_A, {(const cutlass::int4b_t*)B.data_ptr<int8_t>(), ldb},
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

torch::Tensor bmm_s8t_s8n_s32t_cublas(torch::Tensor A, torch::Tensor B) {
  int batch_size = A.size(0);
  int M = A.size(1); // row M K, col k m,At
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

}