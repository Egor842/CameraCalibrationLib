#pragma once
#include "BaseCalibrator.hpp"
#include "BrownConradyModel.hpp"


namespace ccl {


struct BrownConradyCalibrationResult : public CalibrationResult {
    BrownConradyModel get_camera_model(size_t external_vec_idx = 0) const noexcept;
};


class BrownConradyCalibrator : public Calibrator {
public:
    BrownConradyCalibrator(const cv::Size &image_size) : image_size(image_size) {}
    BrownConradyCalibrator(cv::Size &&image_size) : image_size(std::move(image_size)) {}
    BrownConradyCalibrator(size_t width, size_t height) : image_size(width, height) {}

    BrownConradyCalibrationResult calibrate_brown_conrady(
        const std::vector<std::vector<cv::Point3d>> &object_points,
        const std::vector<std::vector<cv::Point2d>> &image_points
    ) const;

protected:
    virtual std::unique_ptr<CalibrationResult> calibrate_impl(
        const std::vector<std::vector<cv::Point3d>> &object_points,
        const std::vector<std::vector<cv::Point2d>> &image_points
    ) const override;

private:
    cv::Size image_size;

    struct InitialGuess {
        cv::Mat camera_matrix;                    // 3x3 intrinsic matrix
        cv::Mat distortion_coeffs;                // 1x5 (k1, k2, p1, p2, k3)
        std::vector<cv::Mat> rotation_vectors;    // rvecs for each view
        std::vector<cv::Mat> translation_vectors; // tvecs for each view
    };

    struct PreparedData {
        cv::Mat all_object_points; // 1 x total_points, CV_64FC3
        cv::Mat all_image_points;  // 1 x total_points, CV_64FC2
        cv::Mat points_per_view;   // 1 x num_views, CV_32SC1
        int total_points;
        int num_views;
    };

    struct IntrinsicInitialGuess {
        cv::Mat camera_matrix; // 3x3
        double fx, fy, cx, cy;
    };

    PreparedData prepare_data(
        const std::vector<std::vector<cv::Point3d>> &object_points,
        const std::vector<std::vector<cv::Point2d>> &image_points
    ) const;

    IntrinsicInitialGuess estimate_intrinsic_matrix(const PreparedData &data) const;

    std::vector<cv::Mat> estimate_extrinsic_parameters(
        const PreparedData &data, const cv::Mat &camera_matrix, const cv::Mat &distortion_coeffs
    ) const;

    std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>> estimate_rotation_and_translation(
        const PreparedData &data, const cv::Mat &camera_matrix, const cv::Mat &distortion_coeffs
    ) const;

    InitialGuess compute_initial_guess(
        const std::vector<std::vector<cv::Point3d>> &object_points,
        const std::vector<std::vector<cv::Point2d>> &image_points
    ) const;
};


}; // namespace ccl
