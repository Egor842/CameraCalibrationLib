#pragma once
#include "../detectors/Pattern.hpp"
#include <fstream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>


namespace ccl::utils {


inline void save_detection_data_yaml(
    const std::string &filename,
    const std::vector<std::optional<cv::Point2d>> &image_points,
    const std::vector<cv::Point3d> &object_points,
    const PatternSize &pattern_size,
    double pattern_step,
    const cv::Size &image_size
) {
    YAML::Emitter out;
    out << YAML::BeginMap;

    out << YAML::Key << "num_image_points" << YAML::Value << image_points.size();
    out << YAML::Key << "image_points" << YAML::Value << YAML::BeginSeq;
    for (const auto &opt : image_points) {
        if (opt.has_value()) {
            const auto &p = *opt;
            out << YAML::Flow << std::vector<double>{p.x, p.y};
        } else {
            out << YAML::Null;
        }
    }
    out << YAML::EndSeq;

    out << YAML::Key << "num_grid_points" << YAML::Value << object_points.size();
    out << YAML::Key << "grid_points" << YAML::Value << YAML::BeginSeq;
    for (const auto &p : object_points) {
        out << YAML::Flow << std::vector<double>{p.x, p.y, p.z};
    }
    out << YAML::EndSeq;

    out << YAML::Key << "board_width" << YAML::Value << static_cast<int>(pattern_size.get_width());
    out << YAML::Key << "board_height" << YAML::Value << static_cast<int>(pattern_size.get_height());
    out << YAML::Key << "board_step" << YAML::Value << pattern_step;
    out << YAML::Key << "image_size" << YAML::Value << YAML::Flow
        << std::vector<int>{image_size.width, image_size.height};

    out << YAML::EndMap;

    std::ofstream fout(filename);
    fout << out.c_str();
}


template <PatternElement T>
void save_detection_data_yaml(
    const std::string &filename,
    const Pattern<T> &pattern,
    const std::vector<cv::Point3d> &object_points,
    double pattern_step,
    const cv::Size &image_size
) {
    const auto &elements = pattern.get_optional_elements();
    std::vector<std::optional<cv::Point2d>> img_pts;
    img_pts.reserve(elements.size());

    for (const auto &opt_elem : elements) {
        if (opt_elem.has_value()) {
            img_pts.emplace_back(opt_elem->major_point());
        } else {
            img_pts.emplace_back(std::nullopt);
        }
    }

    save_detection_data_yaml(filename, img_pts, object_points, pattern.get_size(), pattern_step, image_size);
}


template <PatternElement T>
inline void save_detection_data_yaml(
    const std::string &filename,
    const typename Pattern<T>::ValidMatches &matches,
    const PatternSize &pattern_size,
    double pattern_step,
    const cv::Size &image_size
) {
    std::vector<std::optional<cv::Point2d>> img_pts;
    img_pts.reserve(matches.image_points.size());
    for (const auto &p : matches.image_points) {
        img_pts.emplace_back(p);
    }

    save_detection_data_yaml(filename, img_pts, matches.object_points, pattern_size, pattern_step, image_size);
}


} // namespace ccl::utils
