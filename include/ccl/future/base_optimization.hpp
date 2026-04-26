#pragma once
#include <opencv2/opencv.hpp>
#include <random>


namespace ccl {


class RandomGenerator {
private:
    inline static std::mt19937 gen{std::random_device{}()};
    inline static std::normal_distribution<double> real_normal_dis{0.0, 1.0};
    inline static std::uniform_real_distribution<double> real_uniform_dis{0.0, 1.0};

public:
    static const std::mt19937 &get_gen() noexcept {
        return gen;
    }
    template <typename RandomAccessIterator>
    static void shuffle(RandomAccessIterator first, RandomAccessIterator last) {
        std::shuffle(first, last, gen);
    }

    static int generate_int(int start, int end) noexcept {
        std::uniform_int_distribution<int> dis(start, end);
        return dis(gen);
    }

    static double generate_double(double start, double end) noexcept {
        if (std::abs(start - 0) < 1e-6 && std::abs(end - 0) < 1e-6) {
            return real_uniform_dis(gen);
        }
        std::uniform_real_distribution<double> dis(start, end);
        return dis(gen);
    }

    static double generate_norm_double(double start, double end) noexcept {
        if (std::abs(start - 0) < 1e-6 && std::abs(end - 0) < 1e-6) {
            return real_normal_dis(gen);
        }
        std::normal_distribution<double> dis(start, end);
        return dis(gen);
    }

    static double generate_double() noexcept {
        return real_uniform_dis(gen);
    }
};


inline static bool fill_random_values_lhs(cv::Mat &mat, std::vector<std::pair<double, double>> ranges) {
    if (ranges.empty()) {
        return false;
    }
    if (ranges.size() != mat.cols) {
        return false;
    }
    if (mat.type() != CV_64F) {
        return false;
    }

    const int rows = mat.rows;
    const int cols = mat.cols;

    std::vector<std::vector<double>> lhs_samples(cols, std::vector<double>(rows));

    for (int col = 0; col < cols; ++col) {
        const double range_start = ranges[col].first;
        const double range_end = ranges[col].second;
        const double step = (range_end - range_start) / rows;

        for (int row = 0; row < rows; ++row) {
            double random = RandomGenerator::generate_double(0.0, 1.0);
            lhs_samples[col][row] = range_start + (row + random) * step;
        }

        RandomGenerator::shuffle(lhs_samples[col].begin(), lhs_samples[col].end());
    }

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            mat.at<double>(row, col) = lhs_samples[col][row];
        }
    }

    return true;
}


inline static bool fill_random_values_grid(cv::Mat &mat, std::vector<std::pair<double, double>> ranges) {
    if (ranges.empty()) {
        return false;
    }
    if (ranges.size() != mat.cols) {
        return false;
    }
    if (mat.type() != CV_64F) {
        return false;
    }

    const int rows = mat.rows;
    const int cols = mat.cols;

    for (int idx = 0; idx < cols; ++idx) {
        const auto &range = ranges[idx];
        const double range_start = range.first;
        const double range_end = range.second;
        const double range_size = range_end - range_start;
        const double step = range_size / rows;

        for (int jdx = 0; jdx < rows; ++jdx) {
            double start = range_start + jdx * step;
            double end = (jdx == rows - 1) ? range_end : start + step;
            double random = RandomGenerator::generate_double(0.0, 1.0);
            mat.at<double>(jdx, idx) = start + random * (end - start);
        }
    }

    return true;
}


inline static bool fill_random_values(cv::Mat &mat, std::vector<std::pair<double, double>> ranges) {
    // return fill_random_values_lhs(mat, ranges);
    return fill_random_values_grid(mat, ranges);
}


inline static void fisher_yates_shuffle(std::vector<size_t> &indices, size_t start, size_t end) {
    if (start >= end || end > indices.size()) {
        return;
    }

    for (int idx = static_cast<int>(end) - 1; idx >= static_cast<int>(start); idx--) {
        auto new_idx = RandomGenerator::generate_int(static_cast<int>(start), idx);
        std::swap(indices[new_idx], indices[idx]);
    }
}


inline static void fisher_yates_shuffle(std::vector<size_t> &indices) {
    fisher_yates_shuffle(indices, 0, indices.size());
}


} // namespace ccl
