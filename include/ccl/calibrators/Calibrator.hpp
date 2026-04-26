#pragma once
#include "../models/Distortion.hpp"
#include "CalibrationResult.hpp"
#include <type_traits>


namespace ccl {


enum class LossFunctionType
{
    L2,
    HUBER,
    CAUCHY,
    TUKEY
};


template <typename DistortionType> class ICalibrator {
    static_assert(std::is_base_of_v<Distortion, DistortionType>, "DistortionType must inherit from ccl::Distortion");

protected:
    LossFunctionType loss_function_type = LossFunctionType::CAUCHY;

public:
    virtual ~ICalibrator() = default;

    virtual CalibrationResult<DistortionType> calibrate(
        const std::vector<std::vector<cv::Point3d>> &object_points,
        const std::vector<std::vector<cv::Point2d>> &image_points,
        const cv::Size &image_size
    ) const = 0;
};


}; // namespace ccl
