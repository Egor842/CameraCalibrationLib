#pragma once
#include "base_classes.hpp"
#include <cstdint>
#include <opencv2/opencv.hpp>
#include <optional>


namespace ccl {


struct ChessboardDetectorParams {
    bool norm{false};
    size_t norm_kernel_size{63};
    bool polynomial_fit{true};
    size_t polynomial_fit_kernel_size{9};
    double nms_threshold{0.01};       // base threshold for corner candidates
    double score_threshold{0.01};     // threshold corners score
    bool handle_occlusions{true};     // handle occlusions in board
    bool strict_board_grow{true};     // only precision predictions candidates in board
    bool precision_accuracy{true};
    bool overlay{false};              // allow angles according to national boards
    std::vector<double> scales{};
    std::vector<size_t> radius{5, 7}; // search radii for corner detection
};


class Chessboard : public PatternState {
public:
    Chessboard() = delete;
    Chessboard(
        const std::vector<std::optional<cv::Point2d>> &pixels, const PatternSize &size, const bool full_detected = false
    )
        : PatternState{pixels, size, full_detected} {};
    Chessboard(std::vector<std::optional<cv::Point2d>> &&pixels, PatternSize &&size, bool &&full_detected = false)
        : PatternState{std::move(pixels), std::move(size), full_detected} {};
    ~Chessboard() = default;

    const std::vector<std::optional<cv::Point2d>> &get_pixels_vec() const noexcept {
        return pixels;
    };
    const PatternSize get_size() const noexcept {
        return size;
    }

    std::vector<cv::Point3d> matches_points(const std::vector<cv::Point3d> &points) const noexcept;
};


class ChessboardDetector {
private:
    ChessboardDetectorParams params;
    struct CorrelationTemplates {
        cv::Mat top_right;    // Q1 (+,+)
        cv::Mat top_left;     // Q2 (-,+)
        cv::Mat bottom_left;  // Q3 (-,-)
        cv::Mat bottom_right; // Q4 (+,-)
        size_t template_radius;
        CorrelationTemplates(size_t r, std::pair<double, double> angle_pair);
    };
    struct FilterParams {
        size_t circle_size = 32;
        size_t bin_count = 32;
        size_t cross_threshold = 3;
        size_t target_num_modes = 2;
        size_t target_num_crossing = 4;
    };
    const FilterParams filter_params;
    enum class GrowStatus
    {
        FAILED = 0,
        INTERNAL,
        EXTERNAL,
    };
    mutable std::vector<CorrelationTemplates> templates_vec;
    mutable std::unordered_map<int, cv::Mat> grad_masks;

public:
    struct GradientSet {
        cv::Mat grad_x;
        cv::Mat grad_y;
        cv::Mat angle_img;
        cv::Mat grad_img;

        GradientSet() = default;

        GradientSet(const cv::Size &size, int type)
            : grad_x(size, type),
              grad_y(size, type),
              angle_img(size, type),
              grad_img(size, type) {}

        explicit GradientSet(const cv::Mat &gray);
    };
    struct RawBoard {
        std::vector<std::vector<int>> idx;
        std::vector<std::vector<std::vector<double>>> energy;
        // std::vector<cv::Point2d> pixels;

        int num_corners() const noexcept;
    };
    struct ChessboardCorners {
        std::vector<cv::Point2d> pixels;
        std::vector<size_t> radius;
        std::vector<double> score;
        std::vector<std::array<cv::Point2d, 2>> edge_directions;

        auto edge_first(size_t idx) const {
            return edge_directions[idx][0];
        }
        auto edge_second(size_t idx) const {
            return edge_directions[idx][1];
        }
    };


public:
    ChessboardDetector(const ChessboardDetectorParams &params) : params(params) {
        create_templates_vec();
        create_grad_masks();
        if (this->params.scales.empty()) {
            this->params.scales = {0.5, 1.0, 2.0};
        }
    };
    ChessboardDetector() : params() {
        create_templates_vec();
        create_grad_masks();
        params.scales = {0.5, 1.0, 2.0};
    }
    ~ChessboardDetector() = default;

    [[nodiscard]] std::vector<Chessboard> detect(const cv::Mat &img) const;

private:
    void normalize(cv::Mat &img) const;

    [[nodiscard]] ChessboardCorners get_initial_corners(const cv::Mat &gray) const;

    void create_templates_vec();
    void create_grad_masks();

    [[nodiscard]] std::vector<cv::Point2d>
    non_maximum_suppression(const cv::Mat &gray_image, int n, double tau, int margin) const;

    [[nodiscard]] cv::Mat extract_subpixel_patch(const cv::Mat &img, const cv::Point2d &center, int radius) const;

    [[nodiscard]] cv::Mat
    extract_subpixel_patch(const cv::Mat &img, const cv::Point2d &center, const cv::Mat &mask, int radius) const;

    void raw_refinement(const cv::Mat &gray, const GradientSet &grad_set, ChessboardCorners &corners) const;

    void merge_raw_corners(ChessboardCorners &corners, ChessboardCorners &new_corners, double scale) const;

    [[nodiscard]] std::vector<std::pair<int, double>>
    meanshift(const std::vector<double> &hist, double sigma = 1.5) const;

    [[nodiscard]] ChessboardCorners
    filter_corners(const cv::Mat &gray, const GradientSet &grad_set, ChessboardCorners &corners) const;

    [[nodiscard]] std::pair<cv::Mat, size_t> create_cone_filter(int r) const;

    [[nodiscard]] std::vector<std::array<double, 2>> get_corner_orientation(GradientSet &grad_set) const;

    void refine_corners(GradientSet &grad_set, ChessboardCorners &corners) const;

    [[nodiscard]] std::optional<cv::Point2d> full_refinement_corner(
        const cv::Point2d &p, GradientSet &grad_set, std::vector<std::array<double, 2>> &corner_orientation, double r
    ) const;

    void polynomial_fit(const cv::Mat &gray, ChessboardCorners &corners, int r) const;

    [[nodiscard]] double
    get_corner_score(const cv::Mat &patch, const cv::Mat &grad_patch, std::array<cv::Point2d, 2> &edge_direction) const;

    void calculate_corners_score(const cv::Mat &gray, GradientSet &grad_set, ChessboardCorners &corners) const;

    [[nodiscard]] std::optional<RawBoard>
    init_board(const ChessboardCorners &corners, std::vector<uint8_t> &used, int idx) const;

    [[nodiscard]] int find_neighbor_along_dir(
        const ChessboardCorners &corners, const std::vector<uint8_t> &used, int idx, const cv::Point2d &target_dir
    ) const;

    [[nodiscard]] cv::Point3i board_energy(const ChessboardCorners &corners, RawBoard &board) const;

    [[nodiscard]] double find_minE(const RawBoard &board, const cv::Point2i &p) const;

    void filter_board(
        const ChessboardCorners &corners,
        std::vector<uint8_t> &used,
        RawBoard &board,
        std::vector<cv::Point2i> &proposal,
        double &energy
    ) const;

    [[nodiscard]] GrowStatus grow_board(
        const ChessboardCorners &corners,
        std::vector<uint8_t> &used,
        RawBoard &board,
        std::vector<cv::Point2i> &proposal,
        int direction
    ) const;

    void handle_occlusions(
        const ChessboardCorners &corners,
        RawBoard &board,
        std::vector<cv::Point2i> &proposal,
        std::vector<uint8_t> &used,
        int direction
    ) const;

    [[nodiscard]] bool grow_board_dir(
        const ChessboardCorners &corners,
        RawBoard &board,
        std::vector<cv::Point2i> &proposal,
        std::vector<uint8_t> &used,
        int direction
    ) const;

    [[nodiscard]] bool add_board_bound(RawBoard &board, int direction) const;

    [[nodiscard]] std::vector<int> predict_board_corners(
        const ChessboardCorners &corners,
        std::vector<uint8_t> &used,
        std::vector<int> &p1,
        std::vector<int> &p2,
        std::vector<int> &p3
    ) const;

    [[nodiscard]] std::vector<cv::Point2d> predict_corners(
        const ChessboardCorners &corners,
        const std::vector<int> &far_indices,
        const std::vector<int> &middle_indices,
        const std::vector<int> &near_indices
    ) const;

    [[nodiscard]] std::vector<RawBoard> boards_from_corners(const cv::Mat &img, const ChessboardCorners &corners) const;
};
}; // namespace ccl
