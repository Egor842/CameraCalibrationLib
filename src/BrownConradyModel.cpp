#include "../include/BrownConradyModel.hpp"
#include <fstream>
#include <yaml-cpp/yaml.h>


namespace ccl {


BrownConradyDistortion::BrownConradyDistortion(double k1_, double k2_, double p1_, double p2_, double k3_)
    : k1(k1_),
      k2(k2_),
      p1(p1_),
      p2(p2_),
      k3(k3_) {}


void BrownConradyDistortion::distort(double x, double y, double &xd, double &yd) const noexcept {
    double x2 = x * x;
    double y2 = y * y;
    double r2 = x2 + y2;
    double r4 = r2 * r2;
    double r6 = r4 * r2;

    double radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
    double tx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x2);
    double ty = p1 * (r2 + 2.0 * y2) + 2.0 * p2 * x * y;

    xd = x * radial + tx;
    yd = y * radial + ty;
}


void BrownConradyDistortion::undistort(
    double xd, double yd, double &x, double &y, int max_iter, double eps
) const noexcept {
    x = xd;
    y = yd;

    for (int iter = 0; iter < max_iter; ++iter) {
        double x2 = x * x;
        double y2 = y * y;
        double r2 = x2 + y2;
        double r4 = r2 * r2;
        double r6 = r4 * r2;

        double radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
        double d_radial_dr2 = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4;

        double tx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x2);
        double ty = p1 * (r2 + 2.0 * y2) + 2.0 * p2 * x * y;

        double dtx_dx = 2.0 * p1 * y + p2 * (2.0 * x + 4.0 * x);
        double dtx_dy = 2.0 * p1 * x + p2 * (2.0 * y);
        double dty_dx = p1 * (2.0 * y) + 2.0 * p2 * y;
        double dty_dy = p1 * (2.0 * y + 4.0 * y) + 2.0 * p2 * x;

        double J11 = radial + x * x * d_radial_dr2 * 2.0 + dtx_dx;
        double J12 = x * y * d_radial_dr2 * 2.0 + dtx_dy;
        double J21 = x * y * d_radial_dr2 * 2.0 + dty_dx;
        double J22 = radial + y * y * d_radial_dr2 * 2.0 + dty_dy;

        double det = J11 * J22 - J12 * J21;
        if (std::abs(det) < 1e-12) {
            break;
        }

        double fx_val = x * radial + tx - xd;
        double fy_val = y * radial + ty - yd;

        double dx = (J22 * fx_val - J12 * fy_val) / det;
        double dy = (J11 * fy_val - J21 * fx_val) / det;

        x -= dx;
        y -= dy;

        if (std::abs(dx) < eps && std::abs(dy) < eps) {
            break;
        }
    }
}

std::vector<cv::Point2d> BrownConradyDistortion::distort(const std::vector<cv::Point2d> &points) const noexcept {
    std::vector<cv::Point2d> result;
    result.reserve(points.size());
    for (const auto &p : points) {
        double xd, yd;
        distort(p.x, p.y, xd, yd);
        result.emplace_back(xd, yd);
    }
    return result;
}


std::vector<cv::Point2d> BrownConradyDistortion::undistort(const std::vector<cv::Point2d> &points) const noexcept {
    std::vector<cv::Point2d> result;
    result.reserve(points.size());
    for (const auto &p : points) {
        double x, y;
        undistort(p.x, p.y, x, y);
        result.emplace_back(x, y);
    }
    return result;
}


std::vector<double> BrownConradyDistortion::get_coefficients() const noexcept {
    return {k1, k2, p1, p2, k3};
}


bool BrownConradyDistortion::set_coefficients(const std::vector<double> &coeffs) {
    if (coeffs.size() < 5) {
        return false;
    }
    k1 = coeffs[0];
    k2 = coeffs[1];
    p1 = coeffs[2];
    p2 = coeffs[3];
    k3 = coeffs[4];
    return true;
}


BrownConradyCamera::BrownConradyCamera(
    const IntrisicParams &intrisic, const ExternalParams &external, const BrownConradyDistortion &dist
)
    : CameraModel(intrisic, external, std::make_unique<BrownConradyDistortion>(dist)) {}


BrownConradyCamera::BrownConradyCamera(
    const IntrisicParams &intrinsic,
    const ExternalParams &external,
    double k1,
    double k2,
    double p1,
    double p2,
    double k3
)
    : CameraModel(intrinsic, external, std::make_unique<BrownConradyDistortion>(k1, k2, p1, p2, k3)) {}


std::vector<cv::Point2d> BrownConradyCamera::project_points(const std::vector<cv::Point3d> &points) const noexcept {
    std::vector<cv::Point2d> result;
    if (points.empty()) {
        return result;
    }
    result.reserve(points.size());

    cv::Matx33d R = external.rotation_matrix();
    cv::Vec3d t = external.tvec;
    double fx = intrisic.fx, fy = intrisic.fy;
    double cx = intrisic.cx, cy = intrisic.cy;

    for (const auto &p : points) {
        double Xc = R(0, 0) * p.x + R(0, 1) * p.y + R(0, 2) * p.z + t[0];
        double Yc = R(1, 0) * p.x + R(1, 1) * p.y + R(1, 2) * p.z + t[1];
        double Zc = R(2, 0) * p.x + R(2, 1) * p.y + R(2, 2) * p.z + t[2];

        if (Zc <= 1e-6) {
            result.emplace_back(-1, -1);
            continue;
        }

        double x = Xc / Zc;
        double y = Yc / Zc;

        double xd, yd;
        distortion->distort(x, y, xd, yd);

        double u = fx * xd + cx;
        double v = fy * yd + cy;

        result.emplace_back(u, v);
    }
    return result;
}


void BrownConradyCamera::create_from_yaml(const std::string &yaml_path) {
    try {
        YAML::Node config = YAML::LoadFile(yaml_path);
        if (config["intrinsic"]) {
            auto intr = config["intrinsic"];
            intrisic.fx = intr["fx"].as<double>();
            intrisic.fy = intr["fy"].as<double>();
            intrisic.cx = intr["cx"].as<double>();
            intrisic.cy = intr["cy"].as<double>();
            intrisic.skew = intr["skew"].as<double>(0.0);
        }
        if (config["extrinsic"]) {
            auto ext = config["extrinsic"];
            auto rvec_node = ext["rvec"];
            auto tvec_node = ext["tvec"];
            external.rvec = cv::Vec3d(rvec_node[0].as<double>(), rvec_node[1].as<double>(), rvec_node[2].as<double>());
            external.tvec = cv::Vec3d(tvec_node[0].as<double>(), tvec_node[1].as<double>(), tvec_node[2].as<double>());
        }
        if (config["distortion"]) {
            auto dist = config["distortion"];
            double k1 = dist["k1"].as<double>(0.0);
            double k2 = dist["k2"].as<double>(0.0);
            double p1 = dist["p1"].as<double>(0.0);
            double p2 = dist["p2"].as<double>(0.0);
            double k3 = dist["k3"].as<double>(0.0);
            distortion = std::make_unique<BrownConradyDistortion>(k1, k2, p1, p2, k3);
        } else {
            distortion = std::make_unique<BrownConradyDistortion>();
        }
    } catch (const YAML::Exception &e) { throw std::runtime_error("Failed to load YAML: " + std::string(e.what())); }
}


void BrownConradyCamera::save_to_yaml(const std::string &yaml_path) const {
    YAML::Node config;

    config["intrinsic"]["fx"] = intrisic.fx;
    config["intrinsic"]["fy"] = intrisic.fy;
    config["intrinsic"]["cx"] = intrisic.cx;
    config["intrinsic"]["cy"] = intrisic.cy;
    config["intrinsic"]["skew"] = intrisic.skew;

    YAML::Node rvec_node;
    rvec_node.push_back(external.rvec[0]);
    rvec_node.push_back(external.rvec[1]);
    rvec_node.push_back(external.rvec[2]);
    config["extrinsic"]["rvec"] = rvec_node;

    YAML::Node tvec_node;
    tvec_node.push_back(external.tvec[0]);
    tvec_node.push_back(external.tvec[1]);
    tvec_node.push_back(external.tvec[2]);
    config["extrinsic"]["tvec"] = tvec_node;

    auto coeffs = distortion->get_coefficients();
    if (coeffs.size() >= 5) {
        config["distortion"]["k1"] = coeffs[0];
        config["distortion"]["k2"] = coeffs[1];
        config["distortion"]["p1"] = coeffs[2];
        config["distortion"]["p2"] = coeffs[3];
        config["distortion"]["k3"] = coeffs[4];
    }

    std::ofstream fout(yaml_path);
    if (!fout.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + yaml_path);
    }
    fout << config;
}


void BrownConradyCamera::set_intrinsic(const IntrisicParams &intrinsic) {
    intrisic = intrinsic;
}


void BrownConradyCamera::set_external(const ExternalParams &ext) {
    external = ext;
}


void BrownConradyCamera::set_distortion(const BrownConradyDistortion &dist) {
    distortion = std::make_unique<BrownConradyDistortion>(dist);
}


IntrisicParams &BrownConradyCamera::get_intrinsic_ref() {
    return intrisic;
}


const IntrisicParams &BrownConradyCamera::get_intrinsic_ref() const {
    return intrisic;
}


ExternalParams &BrownConradyCamera::get_external_ref() {
    return external;
}


const ExternalParams &BrownConradyCamera::get_external_ref() const {
    return external;
}


BrownConradyDistortion *BrownConradyCamera::get_distortion_ptr() const {
    return dynamic_cast<BrownConradyDistortion *>(distortion.get());
}


} // namespace ccl
