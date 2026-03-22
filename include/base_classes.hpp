#include <opencv2/opencv.hpp>


namespace ccl {


class PatternSize {
private:
    cv::Size_<size_t> size;

public:
    PatternSize(const cv::Size &size) : PatternSize(size.width, size.height) {}
    PatternSize(const size_t width, const size_t height) : size(width, height) {
        if (size.width < size.height) {
            std::swap(this->size.height, this->size.width);
        }
    }
    PatternSize(const PatternSize &size) = default;
    PatternSize(PatternSize &&size) = default;

    cv::Size value() const noexcept {
        return size;
    }
    size_t get_width() const noexcept {
        return size.width;
    }
    size_t get_height() const noexcept {
        return size.height;
    }
};


struct PatternState {
    std::vector<std::optional<cv::Point2d>> pixels;
    PatternSize size;
    bool full_detected;

    PatternState(
        const std::vector<std::optional<cv::Point2d>> &pixels, const PatternSize &size, const bool full_detected = false
    )
        : pixels(pixels),
          size(size),
          full_detected(full_detected) {}

    PatternState(std::vector<std::optional<cv::Point2d>> &&pixels, PatternSize &&size, bool full_detected = false)
        : pixels(std::move(pixels)),
          size(std::move(size)),
          full_detected(full_detected) {}

    virtual cv::Mat vizualize(const cv::Mat &img) const noexcept {
        cv::Mat result;
        if (img.empty()) {
            return result;
        }

        img.copyTo(result);

        if (result.channels() == 1) {
            cv::cvtColor(result, result, cv::COLOR_GRAY2BGR);
        }

        const int radius = 5;
        const int thickness = 2;
        const cv::Scalar detected_color(0, 255, 0);
        const cv::Scalar text_color(255, 255, 255);
        const double font_scale = 0.4;

        const int board_width = size.get_width();
        const int board_height = size.get_height();

        if (pixels.size() != board_width * board_height) {
            return result;
        }

        for (size_t i = 0; i < pixels.size(); ++i) {
            int row = i / board_width;
            int col = i % board_width;

            const auto &point_opt = pixels[i];

            if (point_opt.has_value()) {
                const cv::Point2d &pt = point_opt.value();
                cv::circle(result, pt, radius, detected_color, thickness);

                std::string idx_str = std::to_string(row) + "," + std::to_string(col);
                cv::putText(
                    result,
                    idx_str,
                    cv::Point(pt.x + radius + 2, pt.y - radius - 2),
                    cv::FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    text_color,
                    1
                );
            }
        }

        std::vector<cv::Point2d> detected_points;
        for (const auto &pt_opt : pixels) {
            if (pt_opt.has_value()) {
                detected_points.push_back(pt_opt.value());
            }
        }

        if (detected_points.size() >= 4) {
            for (int row = 0; row < board_height; ++row) {
                for (int col = 0; col < board_width - 1; ++col) {
                    int idx = row * board_width + col;
                    int next_idx = row * board_width + (col + 1);

                    if (pixels[idx].has_value() && pixels[next_idx].has_value()) {
                        cv::line(result, pixels[idx].value(), pixels[next_idx].value(), cv::Scalar(255, 255, 0), 1);
                    }
                }
            }

            for (int col = 0; col < board_width; ++col) {
                for (int row = 0; row < board_height - 1; ++row) {
                    int idx = row * board_width + col;
                    int next_idx = (row + 1) * board_width + col;

                    if (pixels[idx].has_value() && pixels[next_idx].has_value()) {
                        cv::line(result, pixels[idx].value(), pixels[next_idx].value(), cv::Scalar(255, 255, 0), 1);
                    }
                }
            }
        }

        size_t detected_count = std::count_if(pixels.begin(), pixels.end(), [](const auto &p) {
            return p.has_value();
        });
        size_t total_count = pixels.size();

        std::string stats = "Detected: " + std::to_string(detected_count) + "/" + std::to_string(total_count) + " (" +
                            std::to_string(100.0 * detected_count / total_count) + "%)";
        cv::putText(result, stats, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);

        return result;
    }

    virtual ~PatternState() = default;
};


}; // namespace ccl
