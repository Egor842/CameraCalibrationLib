#pragma once
#include <opencv2/opencv.hpp>


namespace ccl {


class PatternSize {
private:
    cv::Size_<size_t> size;

public:
    PatternSize() = delete;
    explicit PatternSize(const cv::Size_<size_t> &size) : PatternSize(size.width, size.height) {}
    explicit PatternSize(const size_t width, const size_t height) : size(width, height) {
        if (size.width < size.height) {
            std::swap(this->size.height, this->size.width);
        }
    }
    PatternSize(const PatternSize &size) = default;
    PatternSize(PatternSize &&size) = default;
    ~PatternSize() = default;

    PatternSize &operator=(const PatternSize &size) = default;
    PatternSize &operator=(PatternSize &&size) = default;

    operator cv::Size() const noexcept {
        return cv::Size(static_cast<int>(size.width), static_cast<int>(size.height));
    }
    size_t get_width() const noexcept {
        return size.width;
    }
    size_t get_height() const noexcept {
        return size.height;
    }
    size_t area() const noexcept {
        return size.width * size.height;
    }
};


}; // namespace ccl
