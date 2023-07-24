export TORCH_INT_ROOT=$PWD
export CUTLASS_PATH="$TORCH_INT_ROOT/submodules/cutlass"
export CUDA_PATH="/usr/local/cuda"
export PATH="$CUDA_PATH/bin:$PATH"

# CUDA
export CPATH="$CUDA_PATH/include:$CPATH"
export C_INCLUDE_PATH="$CUDA_PATH/include:$C_INCLUDE_PATH"
export CPLUS_INCLUDE_PATH="$CUDA_PATH/include:$CPLUS_INCLUDE_PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$LD_LIBRARY_PATH"

# CUTLASS
export CPATH=$CUTLASS_PATH/tools/util/include:$CUTLASS_PATH/include:$CPATH
export C_INCLUDE_PATH=$CUTLASS_PATH/tools/util/include:$CUTLASS_PATH/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$CUTLASS_PATH/tools/util/include:$CUTLASS_PATH/include:$CPLUS_INCLUDE_PATH


export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONPATH=$TORCH_INT_ROOT/build/lib.linux-x86_64-cpython-38/torch_int:$PYTHONPATH
export PYTHONPATH=/home/yych/anaconda3/envs/int/lib/python3.8/site-packages/torch_int-0.0.0-py3.8-linux-x86_64.egg:$PYTHONPATH