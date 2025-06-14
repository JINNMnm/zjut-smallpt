cmake_minimum_required(VERSION 3.10)
project(cu_openmp_smallpt LANGUAGES CXX CUDA)

# 设置标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# 启用 OpenMP
find_package(OpenMP REQUIRED)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fopenmp")
# 包含头文件路径
include_directories(src)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)


# 源文件
set(SOURCES
    src/kernel.cu
)

# 添加可执行文件
add_executable(cu_openmp_smallpt ${SOURCES})

# 设置 CUDA 编译选项（必须加上 -Xcompiler -fopenmp）
target_compile_options(cu_openmp_smallpt PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>
)

# 链接 CUDA + OpenMP
target_link_libraries(cu_openmp_smallpt
    OpenMP::OpenMP_CXX
    cudart
    curand
)

# CUDA 分离编译（如果需要）
set_target_properties(cu_openmp_smallpt PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)