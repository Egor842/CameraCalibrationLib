#pragma once
#include "BaseCalibrator.hpp"
#include "BrownConradyModel.hpp"


namespace ccl {


struct BrownConradyCalibrationResult : public CalibrationResult {
    BrownConradyModel get_camera_model(size_t external_vec_idx = 0) const noexcept;
};


class BrownConradyCalibrator : public Calibrator {
private:
    cv::Size image_size;

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
};


}; // namespace ccl
