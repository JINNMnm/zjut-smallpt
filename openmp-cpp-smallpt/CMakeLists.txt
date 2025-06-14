cmake_minimum_required(VERSION 3.10)
project(smallpt_openmp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 包含头文件
include_directories(src)

# 查找 OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "✅ OpenMP found: ${OpenMP_CXX_FLAGS}")
endif()

# 添加可执行文件
add_executable(smallpt_openmp
    src/cpp-smallpt.cpp
)

# 链接 OpenMP 库
target_link_libraries(smallpt_openmp PRIVATE OpenMP::OpenMP_CXX)