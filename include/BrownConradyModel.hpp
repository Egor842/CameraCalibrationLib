#pragma once


#include "CameraModel.hpp"
#include <string>
#include <vector>


namespace ccl {


class BrownConradyDistortion : public DistortionModel {
private:
    double k1 = 0.0, k2 = 0.0;
    double p1 = 0.0, p2 = 0.0;
    double k3 = 0.0;

public:
    BrownConradyDistortion() = default;
    BrownConradyDistortion(BrownConradyDistortion &&) = default;
    BrownConradyDistortion(const BrownConradyDistortion &) = default;
    BrownConradyDistortion(double k1_, double k2_, double p1_, double p2_, double k3_ = 0.0);

    void distort(double x, double y, double &xd, double &yd) const noexcept override;
    void
    undistort(double xd, double yd, double &x, double &y, int max_iter = 10, double eps = 1e-8) const noexcept override;
    std::vector<cv::Point2d> distort(const std::vector<cv::Point2d> &points) const noexcept override;
    std::vector<cv::Point2d> undistort(const std::vector<cv::Point2d> &points) const noexcept override;
    std::vector<double> get_coefficients() const noexcept override;
    bool set_coefficients(const std::vector<double> &coeffs) override;
};


class BrownConradyModel : public CameraModel {
public:
    BrownConradyModel(
        const IntrinsicParams &intrisic, const ExtrinsicParams &external, const BrownConradyDistortion &dist
    );
    BrownConradyModel(
        const IntrinsicParams &intrisic, const ExtrinsicParams &external, std::unique_ptr<BrownConradyDistortion> dist
    );
    BrownConradyModel(
        const IntrinsicParams &intrinsic,
        const ExtrinsicParams &external,
        double k1,
        double k2,
        double p1,
        double p2,
        double k3 = 0.0
    );

    std::vector<cv::Point2d> project_points(const std::vector<cv::Point3d> &points) const noexcept override;
    void create_from_yaml(const std::string &yaml_path) override;
    void save_to_yaml(const std::string &yaml_path) const override;

    std::unique_ptr<DistortionModel> get_distortion() const noexcept override;
    bool set_distortion(std::unique_ptr<DistortionModel> dist) override;
};


} // namespace ccl
