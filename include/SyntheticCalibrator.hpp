#pragma once


#include "BaseCalibrator.hpp"
#include "CameraModel.hpp"
#include "base_classes.hpp"
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <thread>
#include <vector>


namespace ccl {


struct BoardParams {
    PatternSize board_size;
    double square_size;
    cv::Point3d origin;

    BoardParams() : board_size(6, 8), square_size(0.03), origin(0, 0, 0) {}
    BoardParams(PatternSize sz, double sq_size, cv::Point3d org = cv::Point3d(0, 0, 0))
        : board_size(sz),
          square_size(sq_size),
          origin(org) {}
    BoardParams(const BoardParams &) = default;

    BoardParams &operator=(const BoardParams &) = default;
};


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


struct GenerationParams {
    std::shared_ptr<CameraModel> camera_model;
    BoardParams board_params{};
    cv::Size image_size = cv::Size(640, 480);

    double noise_sigma = 0.3;
    double outlier_ratio = 0.05;
    double outlier_frame_prob = 0.5;
    double outlier_min_sigma = 5.0;
    double outlier_max_sigma = 15.0;

    int num_frames = 10;
    PoseRange pose_range{};

    bool visualize = false;
    bool save_to_files = false;
    std::string output_dir = ".";

    bool validate(std::string &error_msg) const;
};


struct SynthticCalibrationResult : CalibrationResult {
    double fx_error = 0.0, fy_error = 0.0;
    double cx_error = 0.0, cy_error = 0.0;
    double skew_error = 0.0;
    std::vector<double> dist_coeffs_errors;
    std::vector<std::array<double, 3>> rotation_errors;
    std::vector<std::array<double, 3>> translation_errors;

    SynthticCalibrationResult() = default;

    void assign_base(CalibrationResult &&base);

    void compute_errors(
        const IntrinsicParams &true_intrinsic,
        const std::vector<double> &true_dist,
        const std::vector<ExtrinsicParams> &true_extrinsics
    );
};


class SyntheticCalibratorException : public std::exception {
private:
    std::string message;

public:
    explicit SyntheticCalibratorException(const char *msg) : message(msg) {}
    explicit SyntheticCalibratorException(const std::string &msg) : message(msg) {}

    const char *what() const noexcept override {
        return message.c_str();
    }
};


class SyntheticCalibrator {
public:
    explicit SyntheticCalibrator(const GenerationParams &params);

    SynthticCalibrationResult run(const std::unique_ptr<Calibrator> &calibrator, bool generate_new_data = false);

private:
    void generate_data();

    bool is_grid_convex(const std::vector<cv::Point2d> &points, size_t cols, size_t rows) const noexcept;
    std::vector<cv::Point3d> generate_board_points() const;
    bool generate_valid_pose(cv::Mat &rvec, cv::Mat &tvec, const std::vector<cv::Point3d> &board_points) const;

    std::vector<std::vector<cv::Point3d>> get_object_points_3d() const;
    std::vector<std::vector<cv::Point2d>> get_image_points_2d() const;

    void visualization_thread();
    void save_to_files() const;

    GenerationParams params_;
    std::vector<std::vector<cv::Point3d>> object_points_;
    std::vector<std::vector<cv::Point2d>> ideal_points_;
    std::vector<std::vector<cv::Point2d>> noisy_points_;

    std::vector<cv::Mat> rvecs_true_;
    std::vector<cv::Mat> tvecs_true_;

    std::atomic<bool> stop_visualization_;
    std::atomic<bool> visualization_ready_;
    std::mutex vis_mutex_;
    std::condition_variable vis_cv_;
    std::thread vis_thread_;
};


} // namespace ccl
