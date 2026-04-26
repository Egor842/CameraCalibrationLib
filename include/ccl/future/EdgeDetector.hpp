#pragma once
#include "ChainTree.hpp"
#include <opencv2/opencv.hpp>
#include <stack>


namespace ccl {


enum class GradientOperator
{
    PREWITT3_3 = 0, // [-1, 0, 1], [-1, 0, 1], [-1, 0, 1]
    SOBEL3_3,       // [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]
    SCHARR3_3       // [-3, 0, 3], [-10, 0, 10], [-3, 0, 3]
};


enum class GradientNorm
{
    L1 = 0, // Gx + Gy
    L2      // sqrt(Gx^2 + Gy^2)
};


struct EdgeDetectorParams {
    short gradient_threshold = 16;
    short anchor_threshold = 3;
    int detail_ratio = 1;
    GradientOperator grad_op = GradientOperator::PREWITT3_3;
    GradientNorm grad_norm = GradientNorm::L1;
    double guassian_sigma = 1.0;
    size_t min_path_len = 10;
};


class EdgeDetector {

public:
    using Segment = std::vector<cv::Point2i>;
    using Directions = std::vector<uint8_t>;
    using ChainTreeVec = std::vector<ChainTree>;

protected:
    EdgeDetectorParams params;

private:
    struct StackNode {
        int i, j;
        ChainDirection dir;

        StackNode(int row, int col, ChainDirection direction) : i(row), j(col), dir(direction) {}
    };
    using Stack = std::stack<StackNode>;

public:
    EdgeDetector(const EdgeDetectorParams &params) : params(params) {
        if (this->params.detail_ratio < 1) {
            this->params.detail_ratio = 1;
        }
        if (this->params.anchor_threshold < 0) {
            this->params.anchor_threshold = 0;
        }
        if (this->params.gradient_threshold < 1) {
            this->params.gradient_threshold = 1;
        }
    };
    EdgeDetector() : params() {};
    EdgeDetector(const EdgeDetector &) = default;
    EdgeDetector(EdgeDetector &&) = default;
    virtual ~EdgeDetector() = default;

    [[nodiscard]] virtual std::pair<cv::Mat, std::vector<Segment>> detect(const cv::Mat &input_image) const;

protected:
    [[nodiscard]] std::pair<cv::Mat, Directions> compute_gradients_and_directions(cv::Mat &blurred_image) const;
    [[nodiscard]] std::vector<cv::Point2i>
    find_anchor_points(cv::Mat &grad_img, EdgeDetector::Directions &directions) const;
    [[nodiscard]] std::pair<cv::Mat, std::vector<Segment>> compute_result(
        cv::Mat &grad_img, EdgeDetector::Directions &directions, std::vector<cv::Point2i> anchor_points
    ) const;
    void sort_anchor_points(cv::Mat &grad_img, std::vector<cv::Point2i> &anchor_points) const;
    [[nodiscard]] std::pair<cv::Mat, std::vector<Segment>> build_segments_and_edge_image(
        cv::Mat &grad_img, Directions &directions, std::vector<cv::Point2i> anchor_points
    ) const;
    [[nodiscard]] ChainTreeVec process_tasks_stack(
        cv::Mat &edge_img, cv::Mat &grad_img, Directions &directions, std::vector<cv::Point2i> anchor_points
    ) const;
    [[nodiscard]] std::vector<Segment> build_segments_from_chain_tree(ChainTree &chain_tree, cv::Mat &edge_img) const;
    [[nodiscard]] size_t find_longest_path(const std::shared_ptr<ChainNode> &node) const;
    [[nodiscard]] bool check_dir(uint8_t grad_dir, ChainDirection chain_dir) const;
};


} // namespace ccl
