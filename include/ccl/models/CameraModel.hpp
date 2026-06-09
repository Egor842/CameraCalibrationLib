#pragma once
#include "ExtrinsicParams.hpp"
#include "IntrinsicParams.hpp"
#include <opencv2/opencv.hpp>


namespace ccl {


class CameraModel {
protected:
    IntrinsicParams K;
    ExtrinsicParams pose;

public:
    CameraModel() = default;
    CameraModel(const IntrinsicParams &K, const ExtrinsicParams &pose = {}) : K{K}, pose{pose} {}
    CameraModel(IntrinsicParams &&K, ExtrinsicParams &&pose = {}) : K{std::move(K)}, pose{std::move(pose)} {}
    CameraModel(const CameraModel &) = delete;
    CameraModel(CameraModel &&) = default;
    virtual ~CameraModel() = default;

    CameraModel &operator=(const CameraModel &) = delete;
    CameraModel &operator=(CameraModel &&) = default;

    [[nodiscard]] bool set_pose(const ExtrinsicParams &pose);
    [[nodiscard]] bool set_intrinsic(const IntrinsicParams &K, const cv::Size &img_size = {});
    [[nodiscard]] IntrinsicParams get_intrins() const noexcept;
    [[nodiscard]] ExtrinsicParams get_pose() const noexcept;

    [[nodiscard]] virtual std::vector<cv::Point2d> project(const std::vector<cv::Point3d> &pts) const = 0;
    virtual void create_from_yaml(const std::string &yaml_path) = 0;
    virtual void save_to_yaml(const std::string &yaml_path) const = 0;
};


}; // namespace ccl
