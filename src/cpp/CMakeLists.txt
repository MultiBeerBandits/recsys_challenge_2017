cmake_minimum_required(VERSION 3.5)
project(cpp)

set(CMAKE_CXX_STANDARD 14)

# Find dependencies
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenMP)

# Set OpenMP flags
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Project sources
set(SOURCE_FILES main.cpp src/BPRSlim.cpp include/BPRSlim.h include/Utils.h)
add_executable(testBPR ${SOURCE_FILES})

# Link libraries
target_link_libraries (testBPR Eigen3::Eigen)