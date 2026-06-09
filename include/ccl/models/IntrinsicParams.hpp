#pragma once
#include <opencv4/opencv2/opencv.hpp>


namespace ccl {


struct IntrinsicParams {
public:
    double fx = 0.0, fy = 0.0;
    double cx = 0.0, cy = 0.0;
    double skew = 0.0;

public:
    IntrinsicParams() = default;
    IntrinsicParams(double fx, double fy, double cx, double cy, double skew = 0.0);
    IntrinsicParams(const IntrinsicParams &) = default;
    IntrinsicParams(IntrinsicParams &&) = default;
    ~IntrinsicParams() = default;

    IntrinsicParams &operator=(const IntrinsicParams &) = default;
    IntrinsicParams &operator=(IntrinsicParams &&) = default;
    explicit operator cv::Mat() const;

    [[nodiscard]] cv::Mat to_cv_mat() const;
    [[nodiscard]] bool from_cv_mat(const cv::Mat &k) noexcept;
    [[nodiscard]] bool is_valid(const cv::Size &img_size = {}) const noexcept;
};


}; // namespace ccl
