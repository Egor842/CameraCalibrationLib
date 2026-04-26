#pragma once
#include "../detectors/PatternSize.hpp"
#include "../models/CameraModel.hpp"
#include <opencv2/opencv.hpp>
#include <random>
#include <string>
#include <vector>


namespace ccl {


struct PoseRange {
    double angle_min = -0.5;
    double angle_max = 0.5;
    double dist_min = 0.5;
    double dist_max = 2.0;
    double shift_x_min = -0.5;
    double shift_x_max = 0.5;
    double shift_y_min = -0.5;
    double shift_y_max = 0.5;
};


struct SyntheticData {
    int frame_id = 0;
    cv::Size image_size;
    PatternSize board_size;
    double board_step;

    std::vector<cv::Point3d> object_points;
    std::vector<cv::Point2d> ideal_points;
    std::vector<cv::Point2d> noisy_points;

    ExtrinsicParams true_pose;

    enum class MarkerType
    {
        CROSS,
        CIRCLE,
        SQUARE,
        DIAMOND
    };

    [[nodiscard]] cv::Mat visualize(
        size_t marker_radius = 3,
        size_t marker_thickness = 2,
        MarkerType marker = MarkerType::CIRCLE,
        cv::Scalar marker_color = {0, 255, 0}
    ) const;

    void save_to_yaml(const std::string &filename, bool use_noise_points = true) const;
};


class SyntheticDataGeneratorException : public std::exception {
private:
    std::string message;

public:
    explicit SyntheticDataGeneratorException(const char *msg) : message(msg) {}
    explicit SyntheticDataGeneratorException(const std::string &msg) : message(msg) {}
    const char *what() const noexcept override {
        return message.c_str();
    }
};


class SyntheticDataGenerator {
public:
    struct Config {
        PatternSize board_size{6, 9};
        double board_step{1};
        cv::Point3d board_origin{0.0, 0.0, 0.0};

        cv::Size image_size{640, 480};

        double noise_stddev = 0.3;

        double outlier_ratio = 0.05;
        double outlier_frame_prob = 0.5;
        double outlier_min_amplitude = 3.0;
        double outlier_max_amplitude = 8.0;

        PoseRange pose_range{};
        int num_views = 10;
        bool validate(std::string &error_msg) const;
    } config;

public:
    SyntheticDataGenerator() = delete;
    explicit SyntheticDataGenerator(std::unique_ptr<CameraModel> &&camera);
    SyntheticDataGenerator(const SyntheticDataGenerator &) = delete;
    SyntheticDataGenerator &operator=(const SyntheticDataGenerator &) = delete;
    SyntheticDataGenerator(SyntheticDataGenerator &&) = default;
    SyntheticDataGenerator &operator=(SyntheticDataGenerator &&) = default;
    ~SyntheticDataGenerator() = default;

    std::vector<SyntheticData> generate();

private:
    std::unique_ptr<CameraModel> camera;
    mutable std::mt19937 rng;

    bool is_grid_convex(const std::vector<cv::Point2d> &points, int cols, int rows) const noexcept;
    bool set_new_pose(const ExtrinsicParams &pose, const std::vector<cv::Point3d> &board_pts) const;
    ExtrinsicParams generate_random_pose() const;
    void add_noise_and_outliers(
        const std::vector<cv::Point2d> &ideal, std::vector<cv::Point2d> &noisy, bool has_outliers
    ) const;
    cv::Point2d add_noise(const cv::Point2d &pt) const;
};


} // namespace ccl
