#pragma once
#include "CameraModel.hpp"
#include <memory>


namespace ccl {


struct CalibrationResult {
    IntrinsicParams intrinsic;
    std::vector<ExtrinsicParams> extrinsics_vec;
    std::unique_ptr<DistortionModel> distortion;

    bool successfully = false;
    double rmse = 0.0;
    double time_seconds = 0.0;
    std::vector<double> per_view_errors;
};


class Calibrator {
public:
    virtual ~Calibrator() = default;

    std::unique_ptr<CalibrationResult> calibrate(
        const std::vector<std::vector<cv::Point3d>> &object_points,
        const std::vector<std::vector<cv::Point2d>> &image_points
    ) const {
        return calibrate_impl(object_points, image_points);
    }

protected:
    virtual std::unique_ptr<CalibrationResult> calibrate_impl(
        const std::vector<std::vector<cv::Point3d>> &object_points,
        const std::vector<std::vector<cv::Point2d>> &image_points
    ) const = 0;
};


}; // namespace ccl
