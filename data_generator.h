#pragma once
// data_loader.hpp
// Header file for loading generated clustering data

#ifndef DATA_LOADER_HPP
#define DATA_LOADER_HPP

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <Eigen/Dense>

struct Point3D {
    Eigen::Vector3d position;
    Eigen::Matrix3d covariance;
    int true_cluster_id;    // Ground truth (for validation)
    int assigned_cluster_id; // Result from clustering algorithm
    std::vector<double> cluster_probabilities; // Soft clustering probabilities

    Point3D() : position(Eigen::Vector3d::Zero()),
        covariance(Eigen::Matrix3d::Identity()),
        true_cluster_id(-1),
        assigned_cluster_id(-1) {
    }

    // Compute Mahalanobis distance to another point
    double mahalanobisDistance(const Point3D& other) const {
        Eigen::Vector3d diff = position - other.position;
        Eigen::Matrix3d combined_cov = covariance + other.covariance;

        // Add regularization for numerical stability
        combined_cov += 1e-6 * Eigen::Matrix3d::Identity();

        // Compute inverse using Cholesky decomposition for efficiency
        Eigen::LLT<Eigen::Matrix3d> llt(combined_cov);
        if (llt.info() != Eigen::Success) {
            // Fallback to SVD for badly conditioned matrices
            return diff.norm(); // Euclidean distance as fallback
        }

        Eigen::Vector3d inv_cov_diff = llt.solve(diff);
        double distance_squared = diff.dot(inv_cov_diff);

        return std::sqrt(std::max(0.0, distance_squared));
    }

    // Compute probability density under a Gaussian distribution
    double gaussianProbability(const Eigen::Vector3d& mean,
        const Eigen::Matrix3d& cov) const {
        Eigen::Vector3d diff = position - mean;
        Eigen::Matrix3d total_cov = cov + covariance;

        // Add regularization
        total_cov += 1e-6 * Eigen::Matrix3d::Identity();

        Eigen::LLT<Eigen::Matrix3d> llt(total_cov);
        if (llt.info() != Eigen::Success) {
            return 0.0;
        }

        double det = total_cov.determinant();
        if (det <= 0) return 0.0;

        Eigen::Vector3d inv_cov_diff = llt.solve(diff);
        double exponent = -0.5 * diff.dot(inv_cov_diff);
        double normalization = 1.0 / std::sqrt(std::pow(2 * M_PI, 3) * det);

        return normalization * std::exp(exponent);
    }
};

class DataLoader {
public:
    static std::vector<Point3D> loadDataset(const std::string& filename,
        bool verbose = false) {
        std::vector<Point3D> points;
        std::ifstream file(filename);

        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }

        std::string line;
        int line_number = 0;
        int points_loaded = 0;

        while (std::getline(file, line)) {
            line_number++;

            // Skip empty lines and comments
            if (line.empty() || line[0] == '#') {
                continue;
            }

            std::istringstream iss(line);
            Point3D point;

            // Read position (x, y, z)
            if (!(iss >> point.position(0) >> point.position(1) >> point.position(2))) {
                if (verbose) {
                    std::cerr << "Warning: Skipping malformed line " << line_number << std::endl;
                }
                continue;
            }

            // Read covariance matrix (upper triangular: 6 values)
            double cov_00, cov_01, cov_02, cov_11, cov_12, cov_22;
            if (!(iss >> cov_00 >> cov_01 >> cov_02 >> cov_11 >> cov_12 >> cov_22)) {
                if (verbose) {
                    std::cerr << "Warning: Incomplete covariance at line " << line_number << std::endl;
                }
                continue;
            }

            // Construct symmetric covariance matrix
            point.covariance << cov_00, cov_01, cov_02,
                cov_01, cov_11, cov_12,
                cov_02, cov_12, cov_22;

            // Read ground truth cluster ID
            if (!(iss >> point.true_cluster_id)) {
                point.true_cluster_id = -1; // Unknown
            }

            points.push_back(point);
            points_loaded++;
        }

        file.close();

        if (verbose) {
            std::cout << "Loaded " << points_loaded << " points from " << filename << std::endl;
        }

        return points;
    }

    static void saveResults(const std::vector<Point3D>& points,
        const std::string& filename) {
        std::ofstream file(filename);

        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file for writing: " + filename);
        }

        file << "# Clustering Results\n";
        file << "# Format: x y z assigned_cluster_id true_cluster_id ";
        file << "prob_cluster_0 prob_cluster_1 ...\n";

        for (const auto& point : points) {
            file << point.position(0) << " "
                << point.position(1) << " "
                << point.position(2) << " "
                << point.assigned_cluster_id << " "
                << point.true_cluster_id;

            // Write cluster probabilities if available
            for (double prob : point.cluster_probabilities) {
                file << " " << prob;
            }

            file << "\n";
        }

        file.close();
    }

    static void computeClusteringMetrics(const std::vector<Point3D>& points,
        int num_clusters) {
        // Confusion matrix
        std::vector<std::vector<int>> confusion(num_clusters,
            std::vector<int>(num_clusters, 0));

        int correct = 0;
        int total_clustered = 0;

        for (const auto& point : points) {
            if (point.true_cluster_id >= 0 && point.true_cluster_id < num_clusters &&
                point.assigned_cluster_id >= 0 && point.assigned_cluster_id < num_clusters) {
                confusion[point.true_cluster_id][point.assigned_cluster_id]++;
                total_clustered++;

                if (point.true_cluster_id == point.assigned_cluster_id) {
                    correct++;
                }
            }
        }

        std::cout << "\n=== Clustering Metrics ===" << std::endl;
        std::cout << "Total points clustered: " << total_clustered << std::endl;

        // Print confusion matrix
        std::cout << "\nConfusion Matrix:" << std::endl;
        std::cout << "True\\Assigned\t";
        for (int j = 0; j < num_clusters; ++j) {
            std::cout << "C" << j << "\t";
        }
        std::cout << std::endl;

        for (int i = 0; i < num_clusters; ++i) {
            std::cout << "C" << i << "\t\t";
            for (int j = 0; j < num_clusters; ++j) {
                std::cout << confusion[i][j] << "\t";
            }
            std::cout << std::endl;
        }

        // Compute purity for each cluster
        std::cout << "\nCluster Purities:" << std::endl;
        for (int j = 0; j < num_clusters; ++j) {
            int cluster_size = 0;
            int max_true_cluster = 0;

            for (int i = 0; i < num_clusters; ++i) {
                cluster_size += confusion[i][j];
                max_true_cluster = std::max(max_true_cluster, confusion[i][j]);
            }

            if (cluster_size > 0) {
                double purity = (double)max_true_cluster / cluster_size;
                std::cout << "  Cluster " << j << ": "
                    << std::fixed << std::setprecision(3) << purity
                    << " (" << cluster_size << " points)" << std::endl;
            }
        }

        // Overall accuracy (if we have perfect cluster correspondence)
        if (total_clustered > 0) {
            double accuracy = (double)correct / total_clustered;
            std::cout << "\nNaive Accuracy: " << std::fixed << std::setprecision(3)
                << accuracy << std::endl;
        }
    }

    // Utility function to normalize cluster probabilities
    static void normalizeProbabilities(Point3D& point) {
        double sum = 0.0;
        for (double& prob : point.cluster_probabilities) {
            sum += prob;
        }

        if (sum > 1e-10) {
            for (double& prob : point.cluster_probabilities) {
                prob /= sum;
            }
        }
        else {
            // If all probabilities are nearly zero, assign uniform
            int n = point.cluster_probabilities.size();
            if (n > 0) {
                for (double& prob : point.cluster_probabilities) {
                    prob = 1.0 / n;
                }
            }
        }
    }
};

// Timer utility for performance measurement
class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::string name;

public:
    Timer(const std::string& timer_name = "") : name(timer_name) {
        tic();
    }

    void tic() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    double toc(bool print = false) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>
            (end_time - start_time);
        double seconds = duration.count() / 1e6;

        if (print) {
            if (!name.empty()) {
                std::cout << name << ": ";
            }
            std::cout << seconds << " seconds" << std::endl;
        }

        return seconds;
    }
};

#endif // DATA_LOADER_HPP