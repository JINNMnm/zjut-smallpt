cmake_minimum_required(VERSION 3.10)
project(smallpt_cpu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 包含头文件
include_directories(src)

# 添加可执行文件（你可根据实际路径调整）
add_executable(smallpt_cpu
    src/cpp-smallpt.cpp
)

# 不链接 OpenMP