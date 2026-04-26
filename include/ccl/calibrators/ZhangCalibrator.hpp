#pragma once
#include "../models/BrownConradyDistortion.hpp"
#include "Calibrator.hpp"


namespace ccl {


class ZhangCalibrator : public ICalibrator<BrownConradyDistortion> {
public:
    CalibrationResult<BrownConradyDistortion> calibrate(
        const std::vector<std::vector<cv::Point3d>> &object_points,
        const std::vector<std::vector<cv::Point2d>> &image_points,
        const cv::Size &image_size
    ) const override;

    void set_estimation_k3() noexcept;
    void set_estimation_skew() noexcept;
    void set_loss_function(LossFunctionType loss) noexcept;

    ZhangCalibrator &with_k3(bool value = true) noexcept;
    ZhangCalibrator &with_skew(bool value = true) noexcept;
    ZhangCalibrator &with_loss(LossFunctionType loss) noexcept;

private:
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

    bool estimate_k3 = false;
    bool estimate_skew = false;

private:
    PreparedData prepare_data(
        const std::vector<std::vector<cv::Point3d>> &object_points,
        const std::vector<std::vector<cv::Point2d>> &image_points
    ) const;

    IntrinsicInitialGuess estimate_intrinsic_matrix(const PreparedData &data, const cv::Size &image_size) const;

    std::vector<cv::Mat> estimate_extrinsic_parameters(
        const PreparedData &data, const cv::Mat &camera_matrix, const cv::Mat &distortion_coeffs
    ) const;

    std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>> estimate_rotation_and_translation(
        const PreparedData &data, const cv::Mat &camera_matrix, const cv::Mat &distortion_coeffs
    ) const;

    InitialGuess compute_initial_guess(
        const std::vector<std::vector<cv::Point3d>> &object_points,
        const std::vector<std::vector<cv::Point2d>> &image_points,
        const cv::Size &image_size
    ) const;
};


}; // namespace ccl
