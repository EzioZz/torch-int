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
#include <iomanip>

#define cudaCheckErrors(msg)                                   \
    do {                                                       \
        cudaError_t __err = cudaGetLastError();                \
        if (__err != cudaSuccess) {                            \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                    msg, cudaGetErrorString(__err),            \
                    __FILE__, __LINE__);                       \
            fprintf(stderr, "*** FAILED - ABORTING\n");        \
            exit(1);                                           \
        }                                                      \
    } while (0)

torch::Tensor bmm_s8t_s8n_f32t(torch::Tensor A, torch::Tensor B, float alpha) {
  int batch_size = A.size(0);
  int M = A.size(1);
  int N = B.size(1);
  int K = A.size(2);

  auto C = torch::empty({batch_size, M, N},
                        torch::dtype(torch::kFloat32).device(A.device()));
  int lda = A.size(2); // B M K
  int ldb = B.size(2); // B K N
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
  reg_a.y = (reg_a.y - offset) / scale + offset;
  reg_a.z = (reg_a.z - offset) / scale + offset;
  reg_a.w = (reg_a.w - offset) / scale + offset;

  int4 reg_b;
  reg_b.x = __float2ll_rn(reg_a.x);
  reg_b.y = __float2ll_rn(reg_a.y);
  reg_b.z = __float2ll_rn(reg_a.z);
  reg_b.w = __float2ll_rn(reg_a.w);
  
  int4 reg_c;
  reg_b.x = max(min(7, reg_b.x), -8);
  reg_b.y = max(min(7, reg_b.y), -8);
  reg_b.z = max(min(7, reg_b.z), -8);
  reg_b.w = max(min(7, reg_b.w), -8);

  uint8_t res1 = 0;
  uint8_t res2 = 0;
  res1 = ((reg_b.x & 0x80000000) >> 24)|((reg_b.x & 0x00000007)<<4) | ((reg_b.y & 0x80000000) >> 28)|((reg_b.y & 0x00000007));
  res2 = ((reg_b.z & 0x80000000) >> 24)|((reg_b.z & 0x00000007)<<4) | ((reg_b.w & 0x80000000) >> 28)|((reg_b.w & 0x00000007));
  
  // convert 4 float to 4 int4, then store into 2 uint8
  idx = idx/2;
  dst[idx  ] = res1;
  dst[idx+1] = res2;

  return;
}

// __global__ void quant_s32_s4(int* a, uint8_t* dst, const size_t sizeA, float scale, float offset){
//   int idx = (threadIdx.x + blockIdx.x * blockDim.x)*4; // 每个线程负责 4 个元素
//   float4 reg_a = (reinterpret_cast<int4*>(&a[idx]))[0];
//   reg_a.x = (reg_a.x - offset) / scale + offset;
//   reg_a.y = (reg_a.y - offset) / scale + offset;
//   reg_a.z = (reg_a.z - offset) / scale + offset;
//   reg_a.w = (reg_a.w - offset) / scale + offset;

  
  
//   uint8_t* qInt4_A;
//   cudaMalloc((uint8_t**)&qInt4_A, sizeA*sizeof(uint8_t) / 2);

//   constexpr int blockSize = 256;
//   int gridSize = sizeA / blockSize / 4; // 每个线程负责 4 个 float

//   quant_f32_s4<<<gridSize, blockSize>>>(A.data_ptr<float>(), qInt4_A, sizeA, scale, offset);

//   uint8_t* qInt4_A;
//   cudaMalloc((uint8_t**)&qInt4_A, sizeA*sizeof(uint8_t) / 2);

//   constexpr int blockSize = 256;
//   int gridSize = sizeA / blockSize / 4; // 每个线程负责 4 个 float

//   quant_f32_s4<<<gridSize, blockSize>>>(A.data_ptr<float>(), qInt4_A, sizeA, scale, offset);
// }

void printBits(uint8_t num) {
    for(int i = 7; i >= 0; i--) {
        std::cout << ((num & (1 << i)) ? '1' : '0');
    }
    std::cout << ", ";
}

void benchTestQuantize(torch::Tensor A){
  // fp32 -> int8

  float scale = 1.0;
  float offset = 0.0;

  int batch_size = A.size(0);
  int M = A.size(1);
  int K = A.size(2);

  size_t sizeA = batch_size * M * K;
  uint8_t* qInt4_A = (uint8_t*)malloc(sizeA / 2 * sizeof(uint8_t));
  uint8_t* d_qInt4_A;
  float* fp32_A = (float*)malloc(sizeA * sizeof(float));

  cudaMalloc((uint8_t**)&d_qInt4_A, sizeA*sizeof(uint8_t) / 2);

  constexpr int blockSize = 256;
  int gridSize = sizeA / blockSize / 4; // 每个线程负责 4 个 float

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  
  quant_f32_s4<<<gridSize, blockSize>>>(A.data_ptr<float>(), d_qInt4_A, sizeA, scale, offset);
  cudaDeviceSynchronize();
  cudaCheckErrors("yych: ");

  cudaMemcpy(qInt4_A, d_qInt4_A, sizeA*sizeof(uint8_t) / 2, cudaMemcpyDeviceToHost);
  cudaMemcpy(fp32_A, A.data_ptr<float>(), sizeA*sizeof(float), cudaMemcpyDeviceToHost);


  for (int i = 0; i < 10; i++) {
    printBits(qInt4_A[i]);
  }
  std::cout << std::endl;

  for (int i = 0; i < 10; i++) {
    std::cout << fp32_A[i] << ", ";
  }
  std::cout << std::endl;

  constexpr int n_iter = 1000;
  cudaEventRecord(start);
  for (int i = 0; i < n_iter; i++) {
    quant_f32_s4<<<gridSize, blockSize>>>(A.data_ptr<float>(), d_qInt4_A, sizeA, scale, offset);
  }
  cudaEventRecord(end);  
  cudaDeviceSynchronize();


  float ms;
  cudaEventElapsedTime(&ms, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);

  long long workload = (long long)n_iter * (sizeA * 4 + sizeA/2 * 1); // byte
  double bandwidth = ((double)workload ) / ((double)ms/1e3) / (1e9);
  printf("int4 quantize Performance: %lfGBS\n", bandwidth);

  cudaFree(d_qInt4_A);

}

void bmm_s4t_s4n_f32t_test(uint8_t* qInt4_A, uint8_t* qInt4_B, float* C, int batch_size, int M, int N, int K, int lda, int ldb, int ldc){
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
  float alpha = 1.0;
  Gemm gemm_op;
  typename Gemm::Arguments arguments{
      {M, N, K},      {(const cutlass::int4b_t*)qInt4_A, lda},
      batch_stride_A, {(const cutlass::int4b_t*)qInt4_B, ldb},
      batch_stride_B, {C, ldc},
      batch_stride_C, {C, ldc},
      batch_stride_C, {alpha, 0},
      batch_size};

  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check the problem size is supported or not
  cutlass::Status status = gemm_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    std::cout << "cutlass cannot implement" << std::endl;
    throw std::runtime_error("cutlass cannot implement");
  }

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) {
    std::cout << "cutlass cannot initialize" << std::endl;
    throw std::runtime_error("cutlass cannot initialize");
  }

  status = gemm_op();
  if (status != cutlass::Status::kSuccess) {
    std::cout << "cutlass cannot run" << std::endl;
    throw std::runtime_error("cutlass cannot run");
  }
}


torch::Tensor benchTestBmmInt4(torch::Tensor A, torch::Tensor B, float alpha){
  printf("start bench bmm int4 \n");

  int scale = 1.0;
  int offset = 0.0;

  int batch_size = A.size(0);
  int M = A.size(1);
  int K = A.size(2);
  int N = B.size(2); // B K N
  auto C = torch::empty({batch_size, M, N},
                        torch::dtype(torch::kFloat32).device(A.device()));

  size_t sizeA = batch_size * M * K;
  size_t sizeB = batch_size * N * K;
  size_t sizeC = batch_size * M * N;

  int lda = A.size(2);
  int ldb = B.size(2);
  int ldc = C.size(2);

  prnitf("(%d,%d,%d)\n",M,K,N);
  prnitf("(%d,%d,%d)\n",lda,ldb,ldc);

  uint8_t* d_qInt4_A;
  uint8_t* d_qInt4_B;
  // float* C;
  cudaMalloc((uint8_t**)&d_qInt4_A, sizeA*sizeof(uint8_t) / 2);
  cudaMalloc((uint8_t**)&d_qInt4_A, sizeB*sizeof(uint8_t) / 2);
  
  uint8_t* h_qInt4_A = (uint8_t*)malloc(sizeA*sizeof(uint8_t) / 2);
  uint8_t* h_qInt4_B = (uint8_t*)malloc(sizeB*sizeof(uint8_t) / 2);
  float* fp32_A = (float*)malloc(sizeA*sizeof(float));
  float* fp32_B = (float*)malloc(sizeB*sizeof(float));
  // do quant int4
  // quant_f32_s4(float* a, uint8_t* dst, const size_t sizeA, float scale, float offset);
  quant_f32_s4<<<sizeA/256/4, 256>>>((float*)A.data_ptr(), d_qInt4_A, sizeA, scale, offset);
  quant_f32_s4<<<sizeB/256/4, 256>>>((float*)B.data_ptr(), d_qInt4_A, sizeB, scale, offset);
  cudaDeviceSynchronize();
  cudaCheckErrors("yych: ");


  cudaMemcpy(d_qInt4_A, h_qInt4_A, sizeA*sizeof(uint8_t) / 2, cudaMemcpyDeviceToHost);
  cudaMemcpy(d_qInt4_B, h_qInt4_B, sizeA*sizeof(uint8_t) / 2, cudaMemcpyDeviceToHost);

  cudaMemcpy(fp32_A, A.data_ptr(), sizeA * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(fp32_B, B.data_ptr(), sizeB * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < 10; i++) {
    printBits(h_qInt4_B[i]);
  }
  std::cout << std::endl;

  for (int i = 0; i < 10; i++) {
    std::cout << fp32_B[i] << ", ";
  }
  std::cout << std::endl;

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  bmm_s4t_s4n_f32t_test(d_qInt4_A, d_qInt4_B, (float *)C.data_ptr(), batch_size, M, N, K, lda, ldb, ldc);
  cudaDeviceSynchronize();

  // TODO check correctness
  constexpr int n_iter = 1000;
  cudaEventRecord(start);
  for (int i = 0; i < n_iter; i++) {
    bmm_s4t_s4n_f32t_test(d_qInt4_A, d_qInt4_B, (float *)C.data_ptr(), batch_size, M, N, K, lda, ldb, ldc);
  }
  cudaEventRecord(end);  
  cudaDeviceSynchronize();

  float ms;
  cudaEventElapsedTime(&ms, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);

  long long computeLoad = (2ll * batch_size * M * K * N);
  double gops = (double)n_iter * (double)computeLoad / (1.0*1e9) / ((double)ms / 1000.0); // G
  double tops = gops / 1e3;
  long long memoryLoad = (long long)n_iter * (sizeA * 4 + sizeA/2 * 1); // byte
  double bandwidth = ((double)memoryLoad / (1e9) ) / ((double)ms/1e3) ; // GB/s
  printf("bmm ms: %fms\n", ms);
  printf("bmm Performance: %lfGops\n", gops);
  printf("bmm bandwidth:% lfGBps\n", bandwidth);

  cudaFree(d_qInt4_A);
  cudaFree(d_qInt4_B);

  return C;

}


torch::Tensor bmm_s4t_s4n_f32t(torch::Tensor A, torch::Tensor B, float alpha) {
  // 注意，这里输入矩阵 A，矩阵 B 都还是 float 类型的
  benchTestQuantize(A);
  return benchTestBmmInt4(A, B, alpha);

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