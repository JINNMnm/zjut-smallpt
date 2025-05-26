cmake_minimum_required(VERSION 3.10)
project(cu_smallpt LANGUAGES CXX CUDA)

# 设置 C++ 和 CUDA 标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 包含头文件
include_directories(src)

# 添加 CUDA 源文件
add_executable(cu_smallpt
    src/kernel.cu
)

# 可选：针对某些架构开启优化（如 6.1 是 GTX 10 系列，8.6 是 RTX 30）
# set_target_properties(cu_smallpt PROPERTIES CUDA_ARCHITECTURES "61;86")

# 如果 kernel.cu 里用了 C++ 的函数定义（例如 imageio.hpp）
# 需要加上这行，确保它们都用 C++ 编译
set_target_properties(cu_smallpt PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# 可选 CUDA 编译选项（debugging 或性能优化）
# target_compile_options(cu_smallpt PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)