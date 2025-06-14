cmake_minimum_required(VERSION 3.10)
project(cu_mpi_smallpt LANGUAGES CXX CUDA)

# 设置 C++ 和 CUDA 标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 找到 MPI
find_package(MPI REQUIRED)

# 包含头文件路径
include_directories(src)
include_directories(${MPI_INCLUDE_PATH})

# 添加源文件
set(SOURCES
    src/kernel.cu
)

# 添加可执行文件
add_executable(cu_mpi_smallpt ${SOURCES})

# 设置属性
set_target_properties(cu_mpi_smallpt PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# 链接 MPI 和 CUDA 库
target_link_libraries(cu_mpi_smallpt ${MPI_LIBRARIES} cudart curand)

# 添加 MPI 的头文件路径
target_include_directories(cu_mpi_smallpt PRIVATE ${MPI_INCLUDE_PATH})