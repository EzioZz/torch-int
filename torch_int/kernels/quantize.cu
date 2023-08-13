#include "include/common.h"
#include <cutlass/core_io.h>
#include <cutlass/cutlass.h>
#include <cutlass/integer_subbyte.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/host_tensor.h>


__global__ void float4_add(float* a, float* b, float* c){
    int idx = (threadIdx.x + blockIdx.x * blockDim.x)*4;
    float4 reg_a = (reinterpret_cast<float4*>(&a[idx]))[0];
    float4 reg_b = (reinterpret_cast<float4*>(&b[idx]))[0];
    float4 reg_c;

    reg_c.x = reg_a.x + reg_b.x;
    reg_c.y = reg_a.y + reg_b.y;
    reg_c.z = reg_a.z + reg_b.z;
    reg_c.w = reg_a.w + reg_b.w;

    (reinterpret_cast<float4*>(&c[idx]))[0] = reg_c;
    

}

__global__ void quant_f32_s4(const float* a, uint8_t* dst, const size_t sizeA, float scale, float offset) {

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


torch::Tensor invokeQuantize(torch::Tensor A, float scale, float offset){
    // input = torch.clamp(torch.round(input / qinput_interval - offset) + offset, min_v, max_v)
    int batch_size = A.size(0);
    int M = A.size(1);
    int N = A.size(2);
    constexpr size_t sizeA = batch_size * M * N;
    
    uint8_t* qint4_A;
    cudaMalloc((uint8_t**)&qint4_A, sizeA*sizeof(uint8_t) / 2);

    constexpr int blockSize = 256;
    constexpr int gridSize = sizeA / blockSize / 4; // 每个线程负责 4 个 float

    
    

}

void invokeQuanBmmFused(){





}