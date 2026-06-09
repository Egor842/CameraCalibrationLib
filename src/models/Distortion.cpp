#include "../../include/ccl/models/Distortion.hpp"
#include <execution>


namespace ccl {


std::vector<cv::Point2d> Distortion::undistort(const std::vector<cv::Point2d> &pixels) const noexcept {
    if (pixels.empty()) {
        return {};
    }

    std::vector<cv::Point2d> result(pixels.size());

    std::transform(
        std::execution::par_unseq, pixels.begin(), pixels.end(), result.begin(), [this](const cv::Point2d &p) {
            double x = 0.0, y = 0.0;
            this->undistort(p.x, p.y, x, y);
            return cv::Point2d(x, y);
        }
    );

    return result;
}


std::vector<cv::Point2d> Distortion::distort(const std::vector<cv::Point2d> &pixels) const noexcept {
    if (pixels.empty()) {
        return {};
    }

    std::vector<cv::Point2d> result(pixels.size());

    std::transform(
        std::execution::par_unseq, pixels.begin(), pixels.end(), result.begin(), [this](const cv::Point2d &p) {
            double xd = 0.0, yd = 0.0;
            this->distort(p.x, p.y, xd, yd);
            return cv::Point2d(xd, yd);
        }
    );

    return result;
}


}; // namespace ccl
