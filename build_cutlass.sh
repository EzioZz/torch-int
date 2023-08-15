export CUDACXX=/usr/local/cuda/bin/nvcc
export CC=/usr/gcc84/bin/gcc
export CXX=/usr/gcc84/bin/g++
cd submodules/cutlass
rm -rf build
mkdir -p build && cd build
cmake .. -DCUTLASS_NVCC_ARCHS=80 -DCUTLASS_ENABLE_TESTS=ON -DCUTLASS_UNITY_BUILD_ENABLED=ON
make -j 64
