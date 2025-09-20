# CMakeLists.txt - Advanced version with multiple Eigen acquisition methods
cmake_minimum_required(VERSION 3.14)
project(ClusteringDataGenerator VERSION 1.0.0 LANGUAGES CXX)

# ============================================================================
# Options
# ============================================================================
option(USE_SYSTEM_EIGEN "Try to use system-installed Eigen first" ON)
option(DOWNLOAD_EIGEN "Download Eigen if not found on system" ON)
option(EIGEN_USE_GIT "Download Eigen from Git (slower but gets latest)" OFF)
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
option(ENABLE_OPENMP "Enable OpenMP support" ON)
option(ENABLE_FAST_MATH "Enable fast math optimizations" OFF)
option(BUILD_TESTS "Build test programs" OFF)

# Allow user to specify custom Eigen path
set(EIGEN3_ROOT_DIR "" CACHE PATH "Path to Eigen3 installation")

# ============================================================================
# C++ Standard and Compiler Settings
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for IDE integration
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================================
# Compiler Flags
# ============================================================================
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
    add_compile_options(-march=native)
    
    if(ENABLE_FAST_MATH)
        add_compile_options(-ffast-math)
    endif()
    
    # Debug flags
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -DDEBUG")
    
    # Release flags
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
    
    # Release with debug info
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O2 -g")
    
elseif(MSVC)
    add_compile_options(/W4)
    
    if(ENABLE_FAST_MATH)
        add_compile_options(/fp:fast)
    endif()
    
    # Enable parallel compilation
    add_compile_options(/MP)
endif()

# ============================================================================
# Find or Download Eigen3
# ============================================================================
include(FetchContent)

# Function to download Eigen
function(download_eigen)
    if(EIGEN_USE_GIT)
        message(STATUS "Downloading Eigen from Git repository...")
        FetchContent_Declare(
            Eigen3
            GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
            GIT_TAG 3.4.0
            GIT_SHALLOW TRUE
            GIT_PROGRESS TRUE
        )
    else()
        message(STATUS "Downloading Eigen archive (faster)...")
        FetchContent_Declare(
            Eigen3
            URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
            URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        )
    endif()
    
    # Configure Eigen build options
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    
    FetchContent_MakeAvailable(Eigen3)
    
    # Create alias for consistency
    if(NOT TARGET Eigen3::Eigen)
        add_library(Eigen3::Eigen ALIAS eigen)
    endif()
    
    set(Eigen3_FOUND TRUE PARENT_SCOPE)
    message(STATUS "Eigen3 downloaded and configured successfully")
endfunction()

# Try to find Eigen3
set(Eigen3_FOUND FALSE)

# First, check if user specified a custom path
if(EIGEN3_ROOT_DIR)
    message(STATUS "Looking for Eigen3 in custom path: ${EIGEN3_ROOT_DIR}")
    find_package(Eigen3 3.3 QUIET NO_MODULE PATHS ${EIGEN3_ROOT_DIR} NO_DEFAULT_PATH)
endif()

# If not found and USE_SYSTEM_EIGEN is ON, try system paths
if(NOT Eigen3_FOUND AND USE_SYSTEM_EIGEN)
    message(STATUS "Looking for system-installed Eigen3...")
    find_package(Eigen3 3.3 QUIET NO_MODULE)
endif()

# If still not found and DOWNLOAD_EIGEN is ON, download it
if(NOT Eigen3_FOUND AND DOWNLOAD_EIGEN)
    download_eigen()
elseif(NOT Eigen3_FOUND)
    message(FATAL_ERROR "Eigen3 not found. Enable DOWNLOAD_EIGEN or specify EIGEN3_ROOT_DIR")
endif()

if(Eigen3_FOUND AND NOT DOWNLOAD_EIGEN)
    message(STATUS "Using Eigen3 from: ${Eigen3_DIR}")
endif()

# ============================================================================
# OpenMP Support
# ============================================================================
if(ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
    else()
        message(WARNING "OpenMP not found. Compiling without parallel support.")
    endif()
endif()

# ============================================================================
# Main Executable
# ============================================================================
add_executable(data_generator data_generator.cpp)

# Link libraries
target_link_libraries(data_generator PRIVATE Eigen3::Eigen)

if(OpenMP_CXX_FOUND AND ENABLE_OPENMP)
    target_link_libraries(data_generator PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(data_generator PRIVATE USE_OPENMP)
endif()

# Set properties
set_target_properties(data_generator PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# ============================================================================
# Additional Test Programs (Optional)
# ============================================================================
if(BUILD_TESTS)
    # Simple test to verify Eigen works
    add_executable(test_eigen test_eigen.cpp)
    target_link_libraries(test_eigen PRIVATE Eigen3::Eigen)
    set_target_properties(test_eigen PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    
    # Create test_eigen.cpp if it doesn't exist
    if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_eigen.cpp)
        file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/test_eigen.cpp
"#include <iostream>
#include <Eigen/Dense>

int main() {
    Eigen::Matrix3d m = Eigen::Matrix3d::Random();
    std::cout << \"Eigen test matrix:\\n\" << m << std::endl;
    std::cout << \"Determinant: \" << m.determinant() << std::endl;
    std::cout << \"Eigen is working correctly!\" << std::endl;
    return 0;
}
")
    endif()
endif()

# ============================================================================
# Installation
# ============================================================================
include(GNUInstallDirs)

install(TARGETS data_generator
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES data_loader.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install Python visualization script
install(PROGRAMS visualize_clusters.py
    DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# ============================================================================
# CPack for Creating Packages
# ============================================================================
set(CPACK_PACKAGE_NAME "ClusteringDataGenerator")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "3D Clustering Data Generator with Covariance")
set(CPACK_PACKAGE_VENDOR "ECE Department")
set(CPACK_GENERATOR "TGZ;ZIP")

include(CPack)

# ============================================================================
# Configuration Summary
# ============================================================================
message(STATUS "")
message(STATUS "========== Configuration Summary ==========")
message(STATUS "Project:           ${PROJECT_NAME} v${PROJECT_VERSION}")
message(STATUS "Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler:      ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ standard:      C++${CMAKE_CXX_STANDARD}")

if(Eigen3_FOUND)
    if(EIGEN3_ROOT_DIR)
        message(STATUS "Eigen3:            Custom path")
    elseif(DOWNLOAD_EIGEN)
        message(STATUS "Eigen3:            Downloaded")
    else()
        message(STATUS "Eigen3:            System version")
    endif()
endif()

message(STATUS "Options:")
message(STATUS "  USE_SYSTEM_EIGEN:  ${USE_SYSTEM_EIGEN}")
message(STATUS "  DOWNLOAD_EIGEN:    ${DOWNLOAD_EIGEN}")
message(STATUS "  EIGEN_USE_GIT:     ${EIGEN_USE_GIT}")
message(STATUS "  ENABLE_OPENMP:     ${ENABLE_OPENMP}")
if(ENABLE_OPENMP AND OpenMP_CXX_FOUND)
    message(STATUS "    OpenMP version:  ${OpenMP_CXX_VERSION}")
endif()
message(STATUS "  ENABLE_FAST_MATH:  ${ENABLE_FAST_MATH}")
message(STATUS "  BUILD_TESTS:       ${BUILD_TESTS}")

message(STATUS "Install prefix:    ${CMAKE_INSTALL_PREFIX}")
message(STATUS "==========================================")
message(STATUS "")
message(STATUS "To build: cmake --build . --config Release")
message(STATUS "To install: cmake --install . --prefix /your/path")
message(STATUS "")