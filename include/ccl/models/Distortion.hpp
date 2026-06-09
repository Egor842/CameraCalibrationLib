#pragma once
#include <opencv4/opencv2/opencv.hpp>


namespace ccl {


class Distortion {
public:
    Distortion() = default;
    Distortion(const Distortion &) = default;
    Distortion(Distortion &&) = default;
    virtual ~Distortion() = default;

    Distortion &operator=(const Distortion &) = default;
    Distortion &operator=(Distortion &&) = default;

    [[nodiscard]] virtual std::vector<cv::Point2d>
    undistort(const std::vector<cv::Point2d> &pixels) const noexcept final;
    [[nodiscard]] virtual std::vector<cv::Point2d> distort(const std::vector<cv::Point2d> &pixels) const noexcept final;
    virtual void distort(double x, double y, double &xd, double &yd) const noexcept = 0;
    virtual void undistort(double xd, double yd, double &x, double &y) const noexcept = 0;
    [[nodiscard]] virtual std::vector<double> get_coefficients() const noexcept = 0;
    [[nodiscard]] virtual bool set_coefficients(const std::vector<double> &coeffs) = 0;
};


}; // namespace ccl
