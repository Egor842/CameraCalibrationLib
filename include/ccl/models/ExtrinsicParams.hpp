#pragma once
#include <opencv4/opencv2/opencv.hpp>


namespace ccl {


struct ExtrinsicParams {
public:
    cv::Vec3d rvec;
    cv::Vec3d tvec;

public:
    ExtrinsicParams() : rvec(0, 0, 0), tvec(0, 0, 0) {}
    ExtrinsicParams(const cv::Vec3d &r, const cv::Vec3d &t) : rvec(r), tvec(t) {}
    ExtrinsicParams(cv::Vec3d &&r, cv::Vec3d &&t) : rvec(std::move(r)), tvec(std::move(t)) {}
    ExtrinsicParams(const ExtrinsicParams &) = default;
    ExtrinsicParams(ExtrinsicParams &&) = default;
    ~ExtrinsicParams() = default;

    ExtrinsicParams &operator=(const ExtrinsicParams &) = default;
    ExtrinsicParams &operator=(ExtrinsicParams &&) = default;

    [[nodiscard]] cv::Mat rvec_mat() const;
    [[nodiscard]] cv::Mat tvec_mat() const;
    [[nodiscard]] cv::Matx33d rotation_matrix() const;
    [[nodiscard]] cv::Matx44d to_homogeneous() const;

    [[nodiscard]] ExtrinsicParams compose(const ExtrinsicParams &other) const;
    [[nodiscard]] ExtrinsicParams inverse() const;
    [[nodiscard]] bool is_valid() const;

    [[nodiscard]] bool from_cv_mat(const cv::Mat &R, const cv::Mat &t);
};


}; // namespace ccl
