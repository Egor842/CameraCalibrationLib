#pragma once


#include <memory>
#include <opencv2/opencv.hpp>
#include <optional>


namespace ccl {


class DistortionModel {
public:
    virtual ~DistortionModel() = default;

    virtual void distort(double x, double y, double &xd, double &yd) const noexcept = 0;
    virtual void
    undistort(double xd, double yd, double &x, double &y, int max_iter = 10, double eps = 1e-8) const noexcept = 0;
    virtual std::vector<cv::Point2d> undistort(const std::vector<cv::Point2d> &pixels) const noexcept = 0;
    virtual std::vector<cv::Point2d> distort(const std::vector<cv::Point2d> &pixels) const noexcept = 0;
    virtual std::vector<double> get_coefficients() const noexcept = 0;
    virtual bool set_coefficients(const std::vector<double> &coeffs) = 0;
};


struct IntrisicParams {
    double fx = 0.0, fy = 0.0;
    double cx = 0.0, cy = 0.0;
    double skew = 0.0;

    IntrisicParams() = default;
    IntrisicParams(double fx, double fy, double cx, double cy, double skew);

    cv::Mat to_cv_mat() const;
    explicit operator cv::Mat() const;
    bool from_cv_mat(const cv::Mat &k);
    bool is_valid() const;
};


struct ExternalParams {
    cv::Vec3d rvec;
    cv::Vec3d tvec;

    ExternalParams();
    ExternalParams(const cv::Vec3d &r, const cv::Vec3d &t);
    ExternalParams(cv::Vec3d &&r, cv::Vec3d &&t);

    const double *rvec_data() const;
    const double *tvec_data() const;
    double *rvec_data();
    double *tvec_data();

    cv::Mat rvec_mat() const;
    cv::Mat tvec_mat() const;
    cv::Matx33d rotation_matrix() const;

    ExternalParams compose(const ExternalParams &other) const;
    ExternalParams inverse() const;
    bool is_valid() const;

    static std::optional<ExternalParams> create_from_mat(const cv::Mat &R, const cv::Mat &t);
};


class CameraModel {
protected:
    IntrisicParams intrisic;
    ExternalParams external;
    std::unique_ptr<DistortionModel> distortion;

public:
    CameraModel(
        const IntrisicParams &intrisic, const ExternalParams &external, std::unique_ptr<DistortionModel> &&dist
    );
    virtual ~CameraModel() = default;

    virtual std::vector<cv::Point2d> project_points(const std::vector<cv::Point3d> &points) const noexcept = 0;
    virtual void create_from_yaml(const std::string &yaml_path) = 0;
    virtual void save_to_yaml(const std::string &yaml_path) const = 0;

    IntrisicParams get_inrisic() const noexcept;
    ExternalParams get_external() const noexcept;
    auto get_distortion_coeffs() const noexcept;
};


} // namespace ccl
