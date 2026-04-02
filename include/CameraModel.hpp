#pragma once


#include <memory>
#include <opencv2/opencv.hpp>
#include <optional>


namespace ccl {


class DistortionModel {
public:
    virtual ~DistortionModel() = default;
    DistortionModel() = default;
    DistortionModel(DistortionModel &&) = default;
    DistortionModel(const DistortionModel &) = default;

    virtual void distort(double x, double y, double &xd, double &yd) const noexcept = 0;
    virtual void
    undistort(double xd, double yd, double &x, double &y, int max_iter = 10, double eps = 1e-8) const noexcept = 0;
    virtual std::vector<cv::Point2d> undistort(const std::vector<cv::Point2d> &pixels) const noexcept = 0;
    virtual std::vector<cv::Point2d> distort(const std::vector<cv::Point2d> &pixels) const noexcept = 0;
    virtual std::vector<double> get_coefficients() const noexcept = 0;
    virtual bool set_coefficients(const std::vector<double> &coeffs) = 0;
};


struct IntrinsicParams {
    double fx = 0.0, fy = 0.0;
    double cx = 0.0, cy = 0.0;
    double skew = 0.0;

    IntrinsicParams() = default;
    IntrinsicParams(double fx, double fy, double cx, double cy, double skew = 0.0);

    cv::Mat to_cv_mat() const;
    explicit operator cv::Mat() const;
    bool from_cv_mat(const cv::Mat &k);
    bool is_valid() const;

    double fx_error(const IntrinsicParams &other) const noexcept;
    double fy_error(const IntrinsicParams &other) const noexcept;
    double cx_error(const IntrinsicParams &other) const noexcept;
    double cy_error(const IntrinsicParams &other) const noexcept;
    double skew_error(const IntrinsicParams &other) const noexcept;
};


struct ExtrinsicParams {
    cv::Vec3d rvec;
    cv::Vec3d tvec;

    ExtrinsicParams();
    ExtrinsicParams(const cv::Vec3d &r, const cv::Vec3d &t);
    ExtrinsicParams(cv::Vec3d &&r, cv::Vec3d &&t);

    cv::Vec3d get_rvec() const;
    cv::Vec3d get_tvec() const;

    cv::Mat rvec_mat() const;
    cv::Mat tvec_mat() const;
    cv::Matx33d rotation_matrix() const;

    ExtrinsicParams compose(const ExtrinsicParams &other) const;
    ExtrinsicParams inverse() const;
    bool is_valid() const;

    static std::optional<ExtrinsicParams> create_from_mat(const cv::Mat &R, const cv::Mat &t);

    double rvec_error(const ExtrinsicParams &other) const noexcept;
    double tvec_error(const ExtrinsicParams &other) const noexcept;
};


class CameraModel {
protected:
    IntrinsicParams intrisic;
    ExtrinsicParams external;
    std::unique_ptr<DistortionModel> distortion;

public:
    CameraModel(
        const IntrinsicParams &intrisic, const ExtrinsicParams &external, std::unique_ptr<DistortionModel> dist
    );
    virtual ~CameraModel() = default;

    virtual std::vector<cv::Point2d> project_points(const std::vector<cv::Point3d> &points) const noexcept = 0;
    virtual void create_from_yaml(const std::string &yaml_path) = 0;
    virtual void save_to_yaml(const std::string &yaml_path) const = 0;

    IntrinsicParams get_intrinsic() const noexcept;
    ExtrinsicParams get_external() const noexcept;
    virtual std::unique_ptr<DistortionModel> get_distortion() const noexcept = 0;
    void set_external(const ExtrinsicParams &ext) {
        external = ext;
    }
    void set_intrinsic(const IntrinsicParams &intr) {
        intrisic = intr;
    }
    virtual bool set_distortion(std::unique_ptr<DistortionModel> dist) = 0;
};


} // namespace ccl
