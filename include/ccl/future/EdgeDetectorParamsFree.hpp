#pragma once
#include "EdgeDetector.hpp"


namespace ccl {


class EdgeDetectorParamsFree : public EdgeDetector {
private:
    mutable int Np = 0;
    mutable std::vector<double> H;

public:
    EdgeDetectorParamsFree() : EdgeDetector() {};
    EdgeDetectorParamsFree(const EdgeDetectorParamsFree &) = default;
    EdgeDetectorParamsFree(EdgeDetectorParamsFree &&) = default;
    virtual ~EdgeDetectorParamsFree() = default;

    [[nodiscard]] virtual std::pair<cv::Mat, std::vector<Segment>> detect(const cv::Mat &input_image) const;

private:
    void validate_segments() const;
    void compute_H(const cv::Mat &grad_img) const;
    void
    test_segment(cv::Mat &edge_img, const cv::Mat &grad_img, Segment &segment, int first_index, int second_index) const;
    void extract_new_segments() const;
    [[nodiscard]] double calculate_nfa(double H_value, int len) const;
};


}; // namespace ccl
