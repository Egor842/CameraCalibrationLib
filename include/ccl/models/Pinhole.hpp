#pragma once
#include "CameraModel.hpp"
#include "Distortion.hpp"
#include <exception>
#include <execution>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <type_traits>
#include <yaml-cpp/yaml.h>


namespace ccl {


class PinholeException : public std::runtime_error {
public:
    explicit PinholeException(const std::string &msg) : std::runtime_error(msg) {}
    explicit PinholeException(const char *msg) : std::runtime_error(msg) {}
};


template <typename DistortionType> class Pinhole : public CameraModel {
    static_assert(std::is_base_of_v<Distortion, DistortionType>, "DistortionType must inherit from ccl::Distortion");

protected:
    DistortionType distortion;

public:
    Pinhole() = default;
    Pinhole(const IntrinsicParams &K, const ExtrinsicParams &pose = {}, const DistortionType &dist = {});
    Pinhole(IntrinsicParams &&K, ExtrinsicParams &&pose = {}, DistortionType &&dist = {});
    Pinhole(const Pinhole &) = delete;
    Pinhole(Pinhole &&) = default;
    ~Pinhole() = default;

    Pinhole &operator=(const Pinhole &) = delete;
    Pinhole &operator=(Pinhole &&) = default;

    [[nodiscard]] std::vector<cv::Point2d> project(const std::vector<cv::Point3d> &points) const noexcept;
    void create_from_yaml(const std::string &yaml_path);
    void save_to_yaml(const std::string &yaml_path) const;

    [[nodiscard]] DistortionType get_distortion() const noexcept;
    void set_distortion(const DistortionType &dist);
};


template <typename DistortionType>
Pinhole<DistortionType>::Pinhole(const IntrinsicParams &K, const ExtrinsicParams &pose, const DistortionType &dist)
    : CameraModel{K, pose},
      distortion{dist} {};


template <typename DistortionType>
Pinhole<DistortionType>::Pinhole(IntrinsicParams &&K, ExtrinsicParams &&pose, DistortionType &&dist)
    : CameraModel{std::move(K), std::move(pose)},
      distortion{std::move(dist)} {};


template <typename DistortionType>
std::vector<cv::Point2d> Pinhole<DistortionType>::project(const std::vector<cv::Point3d> &points) const noexcept {
    std::vector<cv::Point2d> result;
    if (points.empty()) {
        return result;
    }
    result.reserve(points.size());
    result.resize(points.size());

    cv::Matx33d R = pose.rotation_matrix();
    cv::Vec3d t = pose.tvec;
    double fx = K.fx, fy = K.fy;
    double cx = K.cx, cy = K.cy;
    double skew = K.skew;

    auto project_point = [R, t, fx, fy, cx, cy, skew, this](const cv::Point3d &p) {
        double Xc = R(0, 0) * p.x + R(0, 1) * p.y + R(0, 2) * p.z + t[0];
        double Yc = R(1, 0) * p.x + R(1, 1) * p.y + R(1, 2) * p.z + t[1];
        double Zc = R(2, 0) * p.x + R(2, 1) * p.y + R(2, 2) * p.z + t[2];

        if (Zc <= 1e-6) {
            return cv::Point2d(-1.0, -1.0);
        }

        double x = Xc / Zc;
        double y = Yc / Zc;

        double xd, yd;
        distortion.distort(x, y, xd, yd);

        double u = fx * xd + skew * yd + cx;
        double v = fy * yd + cy;

        return cv::Point2d(u, v);
    };

    constexpr size_t PARALLEL_THRESHOLD = 100;
    if (points.size() >= PARALLEL_THRESHOLD) {
        std::transform(std::execution::par_unseq, points.begin(), points.end(), result.begin(), project_point);
    } else {
        std::transform(std::execution::seq, points.begin(), points.end(), result.begin(), project_point);
    }

    return result;
}


template <typename DistortionType> void Pinhole<DistortionType>::create_from_yaml(const std::string &yaml_path) {
    try {
        YAML::Node config = YAML::LoadFile(yaml_path);

        if (config["intrinsic"]) {
            auto intr = config["intrinsic"];
            K.fx = intr["fx"].as<double>();
            K.fy = intr["fy"].as<double>();
            K.cx = intr["cx"].as<double>();
            K.cy = intr["cy"].as<double>();
            K.skew = intr["skew"].as<double>(0.0);
        } else {
            throw PinholeException("Missing 'intrinsic' section in YAML");
        }

        if (config["extrinsic"]) {
            auto ext = config["extrinsic"];
            auto rvec_node = ext["rvec"];
            auto tvec_node = ext["tvec"];
            if (rvec_node && rvec_node.size() == 3 && tvec_node && tvec_node.size() == 3) {
                pose.rvec = cv::Vec3d(rvec_node[0].as<double>(), rvec_node[1].as<double>(), rvec_node[2].as<double>());
                pose.tvec = cv::Vec3d(tvec_node[0].as<double>(), tvec_node[1].as<double>(), tvec_node[2].as<double>());
            }
        }

        if (config["distortion"]) {
            auto dist_node = config["distortion"];
            if (dist_node["coefficients"] && dist_node["coefficients"].IsSequence()) {
                std::vector<double> coeffs;
                for (const auto &val : dist_node["coefficients"]) {
                    coeffs.push_back(val.as<double>());
                }
                if (!distortion.set_coefficients(coeffs)) {
                    throw PinholeException("Failed to set distortion coefficients (invalid count or values)");
                }
            } else {
                throw PinholeException("Missing or invalid 'coefficients' sequence in distortion section");
            }
        } else {
            distortion = DistortionType{};
        }
    } catch (const YAML::Exception &e) {
        throw PinholeException("YAML parsing error in " + yaml_path + ": " + e.what());
    }
}


template <typename DistortionType> void Pinhole<DistortionType>::save_to_yaml(const std::string &yaml_path) const {
    YAML::Node config;

    config["intrinsic"]["fx"] = K.fx;
    config["intrinsic"]["fy"] = K.fy;
    config["intrinsic"]["cx"] = K.cx;
    config["intrinsic"]["cy"] = K.cy;
    config["intrinsic"]["skew"] = K.skew;

    YAML::Node rvec_node;
    rvec_node.push_back(pose.rvec[0]);
    rvec_node.push_back(pose.rvec[1]);
    rvec_node.push_back(pose.rvec[2]);
    config["extrinsic"]["rvec"] = rvec_node;

    YAML::Node tvec_node;
    tvec_node.push_back(pose.tvec[0]);
    tvec_node.push_back(pose.tvec[1]);
    tvec_node.push_back(pose.tvec[2]);
    config["extrinsic"]["tvec"] = tvec_node;

    auto coeffs = distortion.get_coefficients();
    YAML::Node coeffs_node;
    for (double c : coeffs) {
        coeffs_node.push_back(c);
    }
    config["distortion"]["coefficients"] = coeffs_node;

    std::ofstream fout(yaml_path);
    if (!fout.is_open()) {
        throw PinholeException("Failed to open file for writing: " + yaml_path);
    }
    fout << config;
}


template <typename DistortionType> DistortionType Pinhole<DistortionType>::get_distortion() const noexcept {
    return distortion;
}


template <typename DistortionType> void Pinhole<DistortionType>::set_distortion(const DistortionType &dist) {
    distortion = dist;
}


} // namespace ccl
