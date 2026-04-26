#pragma once
#include "../detectors/PatternSize.hpp"
#include <opencv2/opencv.hpp>
#include <vector>


namespace ccl::utils {


enum class GridType
{
    CHESSBOARD,
    SYMMETRIC_CIRCLES,
    ASYMMETRIC_CIRCLES
};


inline std::vector<cv::Point3d> generate_grid(
    const PatternSize &size,
    double step,
    const cv::Point3d &origin = {0.0, 0.0, 0.0},
    GridType type = GridType::CHESSBOARD
) {
    std::vector<cv::Point3d> points;
    const size_t cols = size.get_width();
    const size_t rows = size.get_height();
    points.reserve(cols * rows);

    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            double x = origin.x + c * step;
            double y = origin.y + r * step;
            if (type == GridType::ASYMMETRIC_CIRCLES && (r % 2 == 1)) {
                x += step * 0.5;
            }
            points.emplace_back(x, y, origin.z);
        }
    }
    return points;
}


} // namespace ccl::utils
