# 3D Clustering Data Generator

This toolkit generates synthetic 3D point cloud data with measurement uncertainty (covariance matrices) for testing parallel clustering algorithms.

## ?? Key Features

### Automatic Dependency Management
- **No manual Eigen installation needed!** The build system automatically downloads and configures Eigen3 if it's not found on your system
- Works on Windows, Linux, and macOS
- Zero configuration required - just build and run

## Features

- Generates multiple Gaussian clusters in 3D space
- Each point has an associated 3x3 covariance matrix representing measurement uncertainty
- Configurable number of clusters, points, spread, and noise levels
- Includes outlier generation
- Provides ground truth labels for validation
- Visualization support with Python/matplotlib

## Building

### Prerequisites
- C++17 compatible compiler (g++, clang++, or MSVC)
- CMake 3.14+ (will auto-download Eigen if needed)
- Python3 with matplotlib, numpy (optional, for visualization)

### Quick Build (Recommended)

**Linux/macOS:**
```bash
chmod +x build.sh
./build.sh
```

**Windows:**
```cmd
build.bat
```

The build scripts will:
- Check for all requirements
- **Automatically download Eigen3 if not found**
- Configure and build the project
- Report the executable location

### Manual Build with CMake

**Option 1: Basic CMake (auto-downloads Eigen if needed)**
```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

**Option 2: Advanced CMake with options**
```bash
mkdir build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DDOWNLOAD_EIGEN=ON \
  -DENABLE_OPENMP=ON \
  -DBUILD_TESTS=ON
make -j$(nproc)
```

**CMake Options:**
- `USE_SYSTEM_EIGEN`: Try system Eigen first (default: ON)
- `DOWNLOAD_EIGEN`: Auto-download if not found (default: ON)  
- `EIGEN_USE_GIT`: Use Git instead of archive (default: OFF)
- `EIGEN3_ROOT_DIR`: Custom Eigen installation path
- `ENABLE_OPENMP`: Enable parallel features (default: ON)
- `BUILD_TESTS`: Build test programs (default: OFF)

### Direct Compilation (if Eigen is installed)
```bash
g++ -std=c++17 -O3 -I/usr/include/eigen3 data_generator.cpp -o data_generator
```

### Installing Eigen3 Manually (Optional)
The build system will **automatically download Eigen** if not found, but you can install it manually:

```bash
# Ubuntu/Debian
sudo apt-get install libeigen3-dev

# macOS
brew install eigen

# Fedora
sudo dnf install eigen3-devel

# Windows (vcpkg)
vcpkg install eigen3
```

## Usage

### Generate Data

**Interactive mode:**
```bash
./data_generator
# Follow the prompts to enter parameters
```

**Command line mode:**
```bash
./data_generator <num_clusters> <total_points> [output_file] [spread] [noise] [outlier_frac] [seed]

# Example: 5 clusters, 10000 points
./data_generator 5 10000 my_data.txt 5.0 0.5 0.05 42
```

**Parameters:**
- `num_clusters`: Number of clusters (1-50)
- `total_points`: Total number of points to generate
- `output_file`: Output filename (default: clustering_data.txt)
- `spread`: Cluster spread/std deviation (default: 5.0)
- `noise`: Measurement noise level (default: 0.5)
- `outlier_frac`: Fraction of outliers (0-0.5, default: 0.05)
- `seed`: Random seed for reproducibility

### Output Files

The generator creates two files:

1. **Full data file** (`clustering_data.txt`):
   - Format: `x y z cov_00 cov_01 cov_02 cov_11 cov_12 cov_22 true_cluster_id`
   - Contains complete position and covariance information
   - Used by clustering algorithms

2. **Visualization file** (`clustering_data_viz.txt`):
   - Format: `x y z cluster_id max_eigenvalue`
   - Simplified format for quick visualization
   - Contains uncertainty magnitude

### Visualize Data

**Basic visualization:**
```bash
python visualize_clusters.py clustering_data_viz.txt
```

**With covariance ellipsoids:**
```bash
python visualize_clusters.py clustering_data.txt --full --ellipsoids
```

**Save plots:**
```bash
python visualize_clusters.py clustering_data_viz.txt --save output_plot
```

**Options:**
- `--full`: Load full format with covariance matrices
- `--no-3d`: Skip 3D plot
- `--no-2d`: Skip 2D projections
- `--ellipsoids`: Plot covariance ellipsoids
- `--stats`: Print dataset statistics
- `--save`: Save plots to files

### Using gnuplot (alternative visualization):
```bash
gnuplot
gnuplot> set xlabel "X"
gnuplot> set ylabel "Y" 
gnuplot> set zlabel "Z"
gnuplot> splot 'clustering_data_viz.txt' using 1:2:3:4 with points palette pt 7
```

## Loading Data in Your Clustering Implementation

```cpp
#include "data_loader.hpp"

int main() {
    // Load dataset
    auto points = DataLoader::loadDataset("clustering_data.txt", true);
    
    // Access point data
    for(const auto& point : points) {
        Eigen::Vector3d position = point.position;
        Eigen::Matrix3d covariance = point.covariance;
        int ground_truth = point.true_cluster_id;
        
        // Compute Mahalanobis distance between points
        double distance = point.mahalanobisDistance(other_point);
    }
    
    // After clustering, compute metrics
    DataLoader::computeClusteringMetrics(points, num_clusters);
    
    // Save results
    DataLoader::saveResults(points, "results.txt");
}
```

## Dataset Examples

### Small test dataset
```bash
./data_generator 3 1000 test.txt 3.0 0.3 0.02
```

### Medium benchmark
```bash
./data_generator 5 10000 medium.txt 5.0 0.5 0.05
```

### Large parallel processing test
```bash
./data_generator 10 100000 large.txt 7.0 0.8 0.1
```

### High noise/overlap challenge
```bash
./data_generator 8 50000 challenge.txt 3.0 2.0 0.15
```

## Data Format Specification

### Full Format
Each line contains:
1. **Position** (3 floats): x, y, z coordinates
2. **Covariance** (6 floats): Upper triangular part of 3x3 symmetric matrix
   - Order: cov[0,0], cov[0,1], cov[0,2], cov[1,1], cov[1,2], cov[2,2]
3. **Ground Truth** (1 int): True cluster ID (-1 for outliers)

### Mathematical Background

**Mahalanobis Distance** between points with uncertainty:
```
d_M(x_i, x_j) = sqrt((x_i - x_j)^T * (?_i + ?_j)^(-1) * (x_i - x_j))
```

**Gaussian Probability** with measurement uncertainty:
```
P(x | ?, ?_cluster) = N(x; ?, ?_cluster + ?_measurement)
```

## Troubleshooting

### Eigen not found
Specify Eigen path explicitly:
```bash
g++ -std=c++17 -O3 -I/path/to/eigen3 data_generator.cpp -o data_generator
```

### Numerical issues
If you encounter singular matrices:
- Increase measurement noise level
- Decrease cluster spread
- The loader includes regularization (adds 1e-6 to diagonal)

### Memory issues with large datasets
For datasets > 1M points, consider:
- Increasing system memory
- Using memory-mapped files
- Implementing streaming processing

## Performance Tips

1. **Cache-friendly access**: Process points sequentially when possible
2. **Vectorization**: Use Eigen's vectorized operations
3. **Parallel distance computation**: Partition distance matrix by rows
4. **Thread pool**: Reuse threads to avoid creation overhead

## Citation

If using this data generator for academic work, please cite:
```
ECE Graduate Course - Parallel 3D Clustering Assignment
Advanced Parallel Computing, 2024
```

## License

Educational use only. Part of graduate coursework assignment.