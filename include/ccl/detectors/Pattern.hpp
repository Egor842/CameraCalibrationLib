#pragma once
#include "PatternSize.hpp"
#include <concepts>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>


namespace ccl {


struct VisualizationParams {
    enum class MarkerType
    {
        CROSS,
        CIRCLE,
        SQUARE,
        DIAMOND
    };

    MarkerType marker_type = MarkerType::CROSS;
    int marker_size = 5;
    int marker_thickness = 2;
    cv::Scalar marker_color{0, 255, 0};

    int line_thickness = 2;
    cv::Scalar line_color{255, 255, 0};

    bool draw_labels = true;
    double font_scale = 0.5;
    cv::Scalar text_color{255, 255, 255};
    int text_thickness = 1;
};


inline void draw_marker(
    cv::Mat &img, cv::Point2d pt, VisualizationParams::MarkerType type, int size, const cv::Scalar &color, int thickness
) {
    switch (type) {
    case VisualizationParams::MarkerType::CROSS:
        cv::drawMarker(img, pt, color, cv::MARKER_CROSS, size, thickness);
        break;
    case VisualizationParams::MarkerType::CIRCLE:
        cv::circle(img, pt, size, color, thickness);
        break;
    case VisualizationParams::MarkerType::SQUARE:
        cv::rectangle(img, cv::Rect(pt.x - size, pt.y - size, size * 2, size * 2), color, thickness);
        break;
    case VisualizationParams::MarkerType::DIAMOND:
        cv::drawMarker(img, pt, color, cv::MARKER_DIAMOND, size, thickness);
        break;
    }
}


template <typename T>
concept PatternElement = requires(T elem, cv::Mat &img, const VisualizationParams &params) {
    { elem.visualize(img, params) } -> std::same_as<void>;
    { elem.major_point() } -> std::convertible_to<cv::Point2d>;
};


template <typename T>
    requires PatternElement<T>
class Pattern {
private:
    std::vector<std::optional<T>> elements;
    PatternSize size;
    bool full_detected;

public:
    struct ValidMatches {
        std::vector<cv::Point2d> image_points;
        std::vector<cv::Point3d> object_points;
    };

public:
    Pattern() = delete;
    Pattern(const std::vector<std::optional<T>> &elements, const PatternSize &size, const bool full_detected = false)
        : elements(elements),
          size(size),
          full_detected(full_detected) {}
    Pattern(std::vector<std::optional<T>> &&elements, PatternSize &&size, bool full_detected = false) noexcept
        : elements(std::move(elements)),
          size(std::move(size)),
          full_detected(full_detected) {}
    Pattern(const Pattern &) = default;
    Pattern(Pattern &&) = default;
    ~Pattern() = default;

    Pattern &operator=(const Pattern &) = default;
    Pattern &operator=(Pattern &&) = default;

    PatternSize get_size() const noexcept;
    bool is_fully_detected() const noexcept;
    std::vector<T> get_valid_elements() const;
    std::vector<std::optional<T>> get_optional_elements() const;
    ValidMatches get_valid_matches(const std::vector<cv::Point3d> &object_points) const;
    void vizualize(cv::Mat &img, const VisualizationParams &params = {}) const noexcept;
};


template <typename T>
    requires PatternElement<T>
PatternSize Pattern<T>::get_size() const noexcept {
    return size;
}


template <typename T>
    requires PatternElement<T>
bool Pattern<T>::is_fully_detected() const noexcept {
    return full_detected;
}


template <typename T>
    requires PatternElement<T>
std::vector<T> Pattern<T>::get_valid_elements() const {
    std::vector<T> result;
    result.reserve(elements.size());

    for (const auto &opt : elements) {
        if (opt.has_value()) {
            result.push_back(*opt);
        }
    }

    return result;
}


template <typename T>
    requires PatternElement<T>
std::vector<std::optional<T>> Pattern<T>::get_optional_elements() const {
    return elements;
}


template <typename T>
    requires PatternElement<T>
void Pattern<T>::vizualize(cv::Mat &img, const VisualizationParams &params) const noexcept {
    if (img.empty()) {
        return;
    }

    if (img.channels() == 1) {
        cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
    }

    for (const auto &opt : elements) {
        if (opt.has_value()) {
            opt->visualize(img, params);
        }
    }

    if (params.draw_labels) {
        size_t detected_count = 0;
        for (const auto &opt : elements) {
            if (opt.has_value()) {
                ++detected_count;
            }
        }

        size_t total_count = elements.size();

        if (total_count > 0) {
            std::string stats = "Detected: " + std::to_string(detected_count) + "/" + std::to_string(total_count) +
                                " (" + std::to_string(100.0 * detected_count / total_count) + "%)";

            cv::putText(img, stats, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        }
    }
}


template <typename T>
    requires PatternElement<T>
typename Pattern<T>::ValidMatches Pattern<T>::get_valid_matches(const std::vector<cv::Point3d> &object_points) const {
    ValidMatches result;

    if (object_points.size() != elements.size()) {
        return result;
    }

    result.image_points.reserve(elements.size());
    result.object_points.reserve(elements.size());

    for (size_t i = 0; i < elements.size(); ++i) {
        if (elements[i].has_value()) {
            result.image_points.push_back(elements[i]->major_point());
            result.object_points.push_back(object_points[i]);
        }
    }

    return result;
}


}; // namespace ccl
