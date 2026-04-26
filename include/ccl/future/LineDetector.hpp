#pragma once
#include "EdgeDetector.hpp"
#include "NFA.hpp"


namespace ccl {


struct LineDetectorParams {
    double line_error = 1.0;
    int min_line_len = 6;
    double max_error = 1.25;
    double max_distance_between_lines = 6.0;
    void compute_min_line_len(const cv::Size &img_size) {
        double logNT = 2.0 * (log10(static_cast<double>(img_size.width)) + log10(static_cast<double>(img_size.height)));
        min_line_len = static_cast<int>(round((-logNT / log10(0.125)) * 0.5));
    }
};


class Line {
private:
    double A, B, C;
    cv::Point2d start;
    cv::Point2d end;
    size_t len;
    size_t first_segment_pixel_idx;

public:
    Line() = default;
    Line(double A, double B, double C, cv::Point2d start, cv::Point2d end, size_t len, size_t first_segment_pixel_idx);

    double get_A() const noexcept;
    double get_B() const noexcept;
    double get_C() const noexcept;
    cv::Point2d get_start() const noexcept;
    cv::Point2d get_end() const noexcept;
    size_t get_len() const noexcept;
    size_t get_first_segment_pixel() const noexcept;

    void set_A(double new_A) noexcept;
    void set_B(double new_B) noexcept;
    void set_C(double new_C) noexcept;
    void set_start(cv::Point2d new_start) noexcept;
    void set_end(cv::Point2d new_end) noexcept;
    void set_len(size_t new_len) noexcept;
    void set_first_segment_pixel(size_t new_idx) noexcept;

    void update_params(cv::Point2d new_start, cv::Point2d new_end) noexcept;
};


class LineDetector {
public:
    using Segment = std::vector<cv::Point2i>;
    using LineCoeffs = std::array<double, 4>; // A, B, C, error

private:
    LineDetectorParams params;
    EdgeDetectorParams edge_params;
    std::unique_ptr<EdgeDetector> edge_detector;
    mutable std::unique_ptr<NFA> nfa;

public:
    LineDetector() {
        edge_params.gradient_threshold = 36;
        edge_params.anchor_threshold = 8;
        edge_params.grad_op = GradientOperator::SOBEL3_3;
        edge_detector = std::make_unique<EdgeDetector>(edge_params);
    };
    LineDetector(LineDetectorParams &params) : params(params) {
        edge_params.gradient_threshold = 36;
        edge_params.anchor_threshold = 8;
        edge_params.grad_op = GradientOperator::SOBEL3_3;
        edge_detector = std::make_unique<EdgeDetector>(edge_params);
    };

    [[nodiscard]] std::vector<Line> detect(const cv::Mat &img) const;
    [[nodiscard]] std::vector<Line> raw_detect(const cv::Mat &img) const;

private:
    [[nodiscard]] LineCoeffs line_fit(const Segment &segment, size_t it_start, size_t it_end) const noexcept;

    [[nodiscard]] cv::Point2d
    projection_point_to_line(const LineCoeffs &line_coeffs, const cv::Point2i &p, bool norm = true) const noexcept;

    [[nodiscard]] std::vector<Line> split_segment(const Segment &segment) const noexcept;

    [[nodiscard]] std::vector<Line>
    combine_collinear_lines(const Segment &segment, const std::vector<Line> &lines) const;

    [[nodiscard]] double compute_distance_between_lines(
        const Line &line1, const Line &line2, std::function<bool(double, double)> condition
    ) const noexcept;

    [[nodiscard]] bool validate_line_on_segment(const cv::Mat &img, const Line &line, const Segment &segment) const;

    [[nodiscard]] bool validate_line_on_segment_in_rect(
        const cv::Mat &img, const Segment &segment, const Line &line, bool use_rect = false
    ) const;

    [[nodiscard]] std::vector<cv::Point2i>
    enumerate_rect_points(const cv::Point2d &start, const cv::Point2d &end, size_t width = 2) const;
};


}; // namespace ccl