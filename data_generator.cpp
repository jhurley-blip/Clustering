// data_generator.cpp
// Compile: g++ -std=c++17 -O3 -I/usr/include/eigen3 data_generator.cpp -o data_generator

#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

class Point3D {
public:
    Eigen::Vector3d position;
    Eigen::Matrix3d covariance;
    int true_cluster_id;  // Ground truth for validation

    Point3D() : position(Eigen::Vector3d::Zero()),
        covariance(Eigen::Matrix3d::Identity()),
        true_cluster_id(-1) {
    }
};

class ClusterParameters {
public:
    Eigen::Vector3d center;
    Eigen::Matrix3d spread_covariance;  // Controls cluster spread
    double measurement_noise_level;      // Controls point covariance magnitude
    int num_points;

    ClusterParameters() : center(Eigen::Vector3d::Zero()),
        spread_covariance(Eigen::Matrix3d::Identity()),
        measurement_noise_level(0.1),
        num_points(0) {
    }
};

class DataGenerator {
private:
    std::mt19937 rng;
    std::normal_distribution<double> standard_normal;
    std::uniform_real_distribution<double> uniform;

public:
    DataGenerator(unsigned seed = std::chrono::steady_clock::now().time_since_epoch().count())
        : rng(seed), standard_normal(0.0, 1.0), uniform(0.0, 1.0) {
    }

    // Generate a random positive definite covariance matrix
    Eigen::Matrix3d generateRandomCovariance(double min_eigenvalue = 0.01,
        double max_eigenvalue = 1.0,
        double anisotropy = 3.0) {
        // Generate random rotation matrix using QR decomposition
        Eigen::Matrix3d random_matrix;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                random_matrix(i, j) = standard_normal(rng);
            }
        }

        Eigen::HouseholderQR<Eigen::Matrix3d> qr(random_matrix);
        Eigen::Matrix3d Q = qr.householderQ();

        // Generate random eigenvalues
        Eigen::Vector3d eigenvalues;
        eigenvalues(0) = uniform(rng) * (max_eigenvalue - min_eigenvalue) + min_eigenvalue;
        eigenvalues(1) = eigenvalues(0) / (1.0 + uniform(rng) * (anisotropy - 1.0));
        eigenvalues(2) = eigenvalues(1) / (1.0 + uniform(rng) * (anisotropy - 1.0));

        // Construct covariance matrix: Q * Lambda * Q^T
        Eigen::Matrix3d Lambda = eigenvalues.asDiagonal();
        Eigen::Matrix3d covariance = Q * Lambda * Q.transpose();

        // Ensure symmetry (numerical precision)
        covariance = 0.5 * (covariance + covariance.transpose());

        return covariance;
    }

    // Generate cluster centers that are well-separated
    std::vector<Eigen::Vector3d> generateClusterCenters(int num_clusters,
        double space_range = 100.0,
        double min_separation = 10.0) {
        std::vector<Eigen::Vector3d> centers;
        std::uniform_real_distribution<double> pos_dist(-space_range / 2, space_range / 2);

        for (int k = 0; k < num_clusters; ++k) {
            Eigen::Vector3d new_center;
            bool valid = false;
            int attempts = 0;

            while (!valid && attempts < 1000) {
                new_center = Eigen::Vector3d(pos_dist(rng), pos_dist(rng), pos_dist(rng));
                valid = true;

                // Check separation from existing centers
                for (const auto& center : centers) {
                    if ((new_center - center).norm() < min_separation) {
                        valid = false;
                        break;
                    }
                }
                attempts++;
            }

            if (!valid) {
                // Fallback: place on a grid
                double grid_step = space_range / std::cbrt(num_clusters);
                int idx = k;
                int x = idx % (int)std::cbrt(num_clusters);
                int y = (idx / (int)std::cbrt(num_clusters)) % (int)std::cbrt(num_clusters);
                int z = idx / ((int)std::cbrt(num_clusters) * (int)std::cbrt(num_clusters));
                new_center = Eigen::Vector3d(x * grid_step - space_range / 2,
                    y * grid_step - space_range / 2,
                    z * grid_step - space_range / 2);
            }

            centers.push_back(new_center);
        }

        return centers;
    }

    // Generate points for a single cluster
    std::vector<Point3D> generateClusterPoints(const ClusterParameters& params,
        int cluster_id) {
        std::vector<Point3D> points;

        // Compute Cholesky decomposition for sampling from multivariate normal
        Eigen::LLT<Eigen::Matrix3d> llt(params.spread_covariance);
        if (llt.info() != Eigen::Success) {
            std::cerr << "Warning: Cluster covariance not positive definite, using identity" << std::endl;
            llt.compute(Eigen::Matrix3d::Identity());
        }
        Eigen::Matrix3d L = llt.matrixL();

        for (int i = 0; i < params.num_points; ++i) {
            Point3D point;

            // Sample from multivariate normal distribution
            Eigen::Vector3d random_vec;
            random_vec << standard_normal(rng), standard_normal(rng), standard_normal(rng);
            point.position = params.center + L * random_vec;

            // Generate measurement covariance for this point
            point.covariance = generateRandomCovariance(
                params.measurement_noise_level * 0.1,
                params.measurement_noise_level,
                2.0  // Moderate anisotropy
            );

            point.true_cluster_id = cluster_id;
            points.push_back(point);
        }

        return points;
    }

    // Main generation function
    std::vector<Point3D> generateDataset(int num_clusters,
        int total_points,
        double cluster_spread = 5.0,
        double measurement_noise = 0.5,
        double outlier_fraction = 0.05) {
        std::vector<Point3D> all_points;

        // Calculate points per cluster
        int regular_points = total_points * (1.0 - outlier_fraction);
        int outlier_points = total_points - regular_points;

        // Distribute points among clusters (with some variation)
        std::vector<int> points_per_cluster(num_clusters);
        int base_points = regular_points / num_clusters;
        int remaining = regular_points % num_clusters;

        std::uniform_real_distribution<double> variation(0.7, 1.3);
        int total_assigned = 0;

        for (int k = 0; k < num_clusters; ++k) {
            if (k < num_clusters - 1) {
                points_per_cluster[k] = base_points * variation(rng);
                total_assigned += points_per_cluster[k];
            }
            else {
                // Last cluster gets remaining points
                points_per_cluster[k] = regular_points - total_assigned;
            }
        }

        // Generate cluster centers
        auto centers = generateClusterCenters(num_clusters, 50.0, 15.0);

        // Generate points for each cluster
        for (int k = 0; k < num_clusters; ++k) {
            ClusterParameters params;
            params.center = centers[k];
            params.spread_covariance = generateRandomCovariance(
                cluster_spread * 0.5,
                cluster_spread * 2.0,
                3.0
            );
            params.measurement_noise_level = measurement_noise;
            params.num_points = points_per_cluster[k];

            auto cluster_points = generateClusterPoints(params, k);
            all_points.insert(all_points.end(), cluster_points.begin(), cluster_points.end());
        }

        // Add outliers
        std::uniform_real_distribution<double> outlier_pos(-75.0, 75.0);
        for (int i = 0; i < outlier_points; ++i) {
            Point3D outlier;
            outlier.position = Eigen::Vector3d(outlier_pos(rng), outlier_pos(rng), outlier_pos(rng));
            outlier.covariance = generateRandomCovariance(
                measurement_noise * 2.0,
                measurement_noise * 5.0,
                4.0
            );
            outlier.true_cluster_id = -1;  // Mark as outlier
            all_points.push_back(outlier);
        }

        // Shuffle points
        std::shuffle(all_points.begin(), all_points.end(), rng);

        return all_points;
    }

    // Save dataset to file
    void saveDataset(const std::vector<Point3D>& points, const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }

        file << std::fixed << std::setprecision(6);
        file << "# 3D Point Cloud with Covariance Matrices\n";
        file << "# Format: x y z cov_00 cov_01 cov_02 cov_11 cov_12 cov_22 true_cluster_id\n";
        file << "# Number of points: " << points.size() << "\n";

        for (const auto& point : points) {
            // Position
            file << point.position(0) << " "
                << point.position(1) << " "
                << point.position(2) << " ";

            // Covariance (upper triangular, since symmetric)
            file << point.covariance(0, 0) << " "
                << point.covariance(0, 1) << " "
                << point.covariance(0, 2) << " "
                << point.covariance(1, 1) << " "
                << point.covariance(1, 2) << " "
                << point.covariance(2, 2) << " ";

            // Ground truth cluster ID
            file << point.true_cluster_id << "\n";
        }

        file.close();
    }

    // Save in simple XYZ format for visualization
    void saveVisualizationFormat(const std::vector<Point3D>& points, const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }

        file << std::fixed << std::setprecision(6);

        for (const auto& point : points) {
            // Simple format: x y z cluster_id eigenvalue_max
            double max_eigenvalue = point.covariance.eigenvalues().real().maxCoeff();
            file << point.position(0) << " "
                << point.position(1) << " "
                << point.position(2) << " "
                << point.true_cluster_id << " "
                << max_eigenvalue << "\n";
        }

        file.close();
    }

    // Print dataset statistics
    void printStatistics(const std::vector<Point3D>& points, int num_clusters) {
        std::cout << "\n=== Dataset Statistics ===" << std::endl;
        std::cout << "Total points: " << points.size() << std::endl;

        // Count points per cluster
        std::vector<int> cluster_counts(num_clusters + 1, 0);  // +1 for outliers (-1)

        for (const auto& point : points) {
            if (point.true_cluster_id >= 0 && point.true_cluster_id < num_clusters) {
                cluster_counts[point.true_cluster_id]++;
            }
            else if (point.true_cluster_id == -1) {
                cluster_counts[num_clusters]++;  // Outliers
            }
        }

        std::cout << "\nPoints per cluster:" << std::endl;
        for (int k = 0; k < num_clusters; ++k) {
            std::cout << "  Cluster " << k << ": " << cluster_counts[k] << " points" << std::endl;
        }
        if (cluster_counts[num_clusters] > 0) {
            std::cout << "  Outliers: " << cluster_counts[num_clusters] << " points" << std::endl;
        }

        // Compute bounding box
        if (!points.empty()) {
            Eigen::Vector3d min_point = points[0].position;
            Eigen::Vector3d max_point = points[0].position;

            for (const auto& point : points) {
                min_point = min_point.cwiseMin(point.position);
                max_point = max_point.cwiseMax(point.position);
            }

            std::cout << "\nBounding box:" << std::endl;
            std::cout << "  Min: (" << min_point.transpose() << ")" << std::endl;
            std::cout << "  Max: (" << max_point.transpose() << ")" << std::endl;
            std::cout << "  Dimensions: (" << (max_point - min_point).transpose() << ")" << std::endl;
        }

        // Average covariance eigenvalues
        double avg_min_eigenvalue = 0.0;
        double avg_max_eigenvalue = 0.0;

        for (const auto& point : points) {
            Eigen::Vector3d eigenvalues = point.covariance.eigenvalues().real();
            avg_min_eigenvalue += eigenvalues.minCoeff();
            avg_max_eigenvalue += eigenvalues.maxCoeff();
        }

        avg_min_eigenvalue /= points.size();
        avg_max_eigenvalue /= points.size();

        std::cout << "\nMeasurement uncertainty (covariance eigenvalues):" << std::endl;
        std::cout << "  Average min: " << avg_min_eigenvalue << std::endl;
        std::cout << "  Average max: " << avg_max_eigenvalue << std::endl;
        std::cout << "  Average condition number: " << avg_max_eigenvalue / avg_min_eigenvalue << std::endl;
    }
};

int main(int argc, char* argv[]) {
    // Default parameters
    int num_clusters = 5;
    int total_points = 1000;
    double cluster_spread = 5.0;
    double measurement_noise = 0.5;
    double outlier_fraction = 0.05;
    std::string output_file = "clustering_data.txt";
    std::string viz_file = "clustering_viz.txt";
    unsigned seed = std::chrono::steady_clock::now().time_since_epoch().count();

    // Parse command line arguments
    std::cout << "=== 3D Clustering Data Generator ===" << std::endl;

    if (argc >= 3) {
        num_clusters = std::atoi(argv[1]);
        total_points = std::atoi(argv[2]);

        if (argc >= 4) output_file = argv[3];
        if (argc >= 5) cluster_spread = std::atof(argv[4]);
        if (argc >= 6) measurement_noise = std::atof(argv[5]);
        if (argc >= 7) outlier_fraction = std::atof(argv[6]);
        if (argc >= 8) seed = std::atoi(argv[7]);
    }
    else {
        // Interactive mode
        std::cout << "\nEnter parameters (or press Enter for defaults):" << std::endl;

        std::string input;
        std::cout << "Number of clusters [" << num_clusters << "]: ";
        std::getline(std::cin, input);
        if (!input.empty()) num_clusters = std::stoi(input);

        std::cout << "Total number of points [" << total_points << "]: ";
        std::getline(std::cin, input);
        if (!input.empty()) total_points = std::stoi(input);

        std::cout << "Output filename [" << output_file << "]: ";
        std::getline(std::cin, input);
        if (!input.empty()) output_file = input;

        std::cout << "Cluster spread (std dev) [" << cluster_spread << "]: ";
        std::getline(std::cin, input);
        if (!input.empty()) cluster_spread = std::stod(input);

        std::cout << "Measurement noise level [" << measurement_noise << "]: ";
        std::getline(std::cin, input);
        if (!input.empty()) measurement_noise = std::stod(input);

        std::cout << "Outlier fraction [" << outlier_fraction << "]: ";
        std::getline(std::cin, input);
        if (!input.empty()) outlier_fraction = std::stod(input);

        std::cout << "Random seed [" << seed << "]: ";
        std::getline(std::cin, input);
        if (!input.empty()) seed = std::stoul(input);
    }

    // Validate parameters
    if (num_clusters < 1 || num_clusters > 50) {
        std::cerr << "Error: Number of clusters must be between 1 and 50" << std::endl;
        return 1;
    }

    if (total_points < num_clusters) {
        std::cerr << "Error: Total points must be at least equal to number of clusters" << std::endl;
        return 1;
    }

    if (outlier_fraction < 0.0 || outlier_fraction > 0.5) {
        std::cerr << "Error: Outlier fraction must be between 0.0 and 0.5" << std::endl;
        return 1;
    }

    // Generate dataset
    std::cout << "\nGenerating dataset..." << std::endl;
    std::cout << "  Clusters: " << num_clusters << std::endl;
    std::cout << "  Total points: " << total_points << std::endl;
    std::cout << "  Cluster spread: " << cluster_spread << std::endl;
    std::cout << "  Measurement noise: " << measurement_noise << std::endl;
    std::cout << "  Outlier fraction: " << outlier_fraction << std::endl;
    std::cout << "  Random seed: " << seed << std::endl;

    try {
        DataGenerator generator(seed);
        auto points = generator.generateDataset(
            num_clusters,
            total_points,
            cluster_spread,
            measurement_noise,
            outlier_fraction
        );

        // Save dataset
        std::cout << "\nSaving dataset..." << std::endl;
        generator.saveDataset(points, output_file);
        std::cout << "  Full data saved to: " << output_file << std::endl;

        // Extract base filename
        size_t lastdot = output_file.find_last_of(".");
        std::string base_name = (lastdot == std::string::npos) ? output_file : output_file.substr(0, lastdot);
        viz_file = base_name + "_viz.txt";

        generator.saveVisualizationFormat(points, viz_file);
        std::cout << "  Visualization format saved to: " << viz_file << std::endl;

        // Print statistics
        generator.printStatistics(points, num_clusters);

        std::cout << "\nDataset generation complete!" << std::endl;

        // Print usage hint
        std::cout << "\nTo visualize with gnuplot:" << std::endl;
        std::cout << "  gnuplot> splot '" << viz_file << "' using 1:2:3:4 with points palette pt 7" << std::endl;

        std::cout << "\nTo load in your clustering program:" << std::endl;
        std::cout << "  std::ifstream file(\"" << output_file << "\");" << std::endl;
        std::cout << "  // Skip comment lines starting with #" << std::endl;
        std::cout << "  // Read: x y z cov(6 values) true_cluster_id" << std::endl;

    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}