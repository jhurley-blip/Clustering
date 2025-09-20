# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(ClusteringDataGenerator)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen3
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")

# Data generator executable
add_executable(data_generator data_generator.cpp)
target_link_libraries(data_generator Eigen3::Eigen)

# Optional: Add OpenMP support for future clustering implementations
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(data_generator OpenMP::OpenMP_CXX)
endif()

# Installation rules
install(TARGETS data_generator DESTINATION bin)