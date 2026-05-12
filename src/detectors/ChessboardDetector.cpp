#include "../../include/ccl/detectors/ChessboardDetector.hpp"
#include <cstdint>
#include <opencv2/core/hal/hal.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <random>
#include <utility>


namespace ccl {


constexpr double EPSILON = 1e-6;
constexpr double DOUBLE_MAX = std::numeric_limits<double>::max();


cv::Point2d ChessboardCorner::major_point() const noexcept {
    return major_pt;
}


void ChessboardCorner::visualize(cv::Mat &img, const VisualizationParams &params) const {
    draw_marker(img, major_pt, params.marker_type, params.marker_size, params.marker_color, params.marker_thickness);

    if (params.draw_labels) {
        std::string label = std::to_string(row) + "," + std::to_string(col);

        cv::Point text_pos(
            static_cast<int>(major_pt.x + params.marker_size + 2), static_cast<int>(major_pt.y - params.marker_size - 2)
        );

        cv::putText(
            img, label, text_pos, cv::FONT_HERSHEY_SIMPLEX, params.font_scale, params.text_color, params.text_thickness
        );
    }
}


int ChessboardDetector::RawBoard::num_corners() const noexcept {
    int num = 0;
    for (const auto &row : idx) {
        for (int jdx : row) {
            if (jdx != -2) {
                num++;
            }
        }
    }
    return num;
}


ChessboardDetector::GradientSet::GradientSet(const cv::Mat &gray) : GradientSet(gray.size(), gray.type()) {
    cv::Sobel(gray, grad_x, CV_64F, 1, 0, 3, 1.0, 0.0, cv::BORDER_REFLECT);
    cv::Sobel(gray, grad_y, CV_64F, 0, 1, 3, 1.0, 0.0, cv::BORDER_REFLECT);

    auto create_img_memory_continuous = [](cv::Mat &img) {
        if (!img.isContinuous()) {
            cv::Mat tmp = img.clone();
            std::swap(tmp, img);
        }
    };

    create_img_memory_continuous(grad_x);
    create_img_memory_continuous(grad_y);
    create_img_memory_continuous(angle_img);
    create_img_memory_continuous(grad_img);

    cv::hal::fastAtan64f(
        (const double *)grad_y.data, (const double *)grad_x.data, (double *)angle_img.data, gray.rows * gray.cols, false
    );
    angle_img.forEach<double>([](double &pixel, const int *pos) -> void {
        pixel = pixel >= M_PI ? pixel - M_PI : pixel;
    });
    grad_img.forEach<double>([this](double &magnitude, const int *position) {
        int col = position[1];
        int row = position[0];

        double gx = grad_x.at<double>(row, col);
        double gy = grad_y.at<double>(row, col);

        magnitude = std::sqrt(gx * gx + gy * gy);
    });
}


// for debug
void plot_corners(const cv::Mat &img, const std::vector<cv::Point2d> &corners, const char *str) {
    cv::Mat img_show;
    if (img.channels() != 3) {
        cv::cvtColor(img, img_show, cv::COLOR_GRAY2BGR);
    } else {
        img_show = img.clone();
    }
    for (int i = 0; i < corners.size(); ++i) {
        cv::circle(img_show, corners[i], 3, cv::Scalar(0, 0, 255), -1);
    }
    cv::resize(img_show, img_show, cv::Size(640 * 2, 480 * 2));
    cv::imshow(str, img_show);
    cv::waitKey(0);
}


// for debug
void visualize_corners_with_directions(
    const cv::Mat &img, const ChessboardDetector::ChessboardCorners &corners, const std::string &winname
) {
    cv::Mat img_show;
    if (img.channels() == 3) {
        img_show = img.clone();
    } else {
        cv::cvtColor(img, img_show, cv::COLOR_GRAY2BGR);
    }

    for (size_t i = 0; i < corners.pixels.size(); ++i) {
        cv::Point2d pt = corners.pixels[i];
        double r = corners.radius[i];
        cv::Point2d dir1 = corners.edge_directions[i][0];
        cv::Point2d dir2 = corners.edge_directions[i][1];
        double len = r * 1.5;
        cv::Point2d end1 = pt + dir1 * len;
        cv::Point2d end2 = pt + dir2 * len;
        cv::line(img_show, pt, end1, cv::Scalar(0, 0, 255), 1);
        cv::line(img_show, pt, end2, cv::Scalar(0, 255, 0), 1);
        cv::circle(img_show, pt, 2, cv::Scalar(255, 0, 0), -1);
    }
    cv::resize(img_show, img_show, cv::Size(640 * 2, 480 * 2));
    cv::imshow(winname, img_show);
    cv::waitKey(0);
}


std::vector<Chessboard> ChessboardDetector::detect(const cv::Mat &img) const {
    cv::Mat gray;
    if (img.channels() == 3) {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        gray.convertTo(gray, CV_64F, 1.0 / 255.0, 0);
    } else {
        img.convertTo(gray, CV_64F, 1.0 / 255.0, 0);
    }

    normalize(gray);

    std::vector<std::pair<cv::Mat, double>> gray_images_pyramid{std::pair(gray, 1.0)};
    for (const auto &scale : params.scales) {
        if (std::abs(scale - 1.0) < EPSILON) {
            continue;
        }
        cv::Mat gray_scale;
        cv::resize(gray, gray_scale, cv::Size(gray.cols * scale, gray.rows * scale), 0, 0, cv::INTER_LINEAR);
        gray_images_pyramid.emplace_back(gray_scale, scale);
    }

    ChessboardCorners corners;
    GradientSet gradient_set(gray);

    for (auto &pyramid_element : gray_images_pyramid) {
        cv::Mat gray_image = pyramid_element.first;
        double scale = pyramid_element.second;

        GradientSet curr_gradient_set;
        if (std::abs(scale - 1.0) < EPSILON) {
            curr_gradient_set = gradient_set;
        } else {
            curr_gradient_set = std::move(GradientSet(gray_image));
        }

        auto new_corners = get_initial_corners(gray_image);

        raw_refinement(gray_image, curr_gradient_set, new_corners);

        new_corners = std::move(filter_corners(gray_image, curr_gradient_set, new_corners));

        refine_corners(curr_gradient_set, new_corners);

        merge_raw_corners(corners, new_corners, scale);
    }

    if (params.polynomial_fit == true) {
        polynomial_fit(gray, corners, params.polynomial_fit_kernel_size / 2);
    }

    calculate_corners_score(gray, gradient_set, corners);

    auto raw_boards = std::move(boards_from_corners(gray, corners));
    std::vector<Chessboard> chessboards;

    // cv::Mat display = img.clone();
    // plot_corners(display, corners.pixels, "g");
    // display = img.clone();
    // visualize_corners_with_directions(display, corners, "gg");
    for (auto &board : raw_boards) {
        int offset = 1;
        int real_height = board.idx.size() - 2 * offset;
        int real_width = board.idx[0].size() - 2 * offset;

        std::vector<std::optional<ChessboardCorner>> corners_vec;
        corners_vec.reserve(real_height * real_width);
        bool full_detected = true;
        ChessboardCorner corner;

        for (int y = offset; y <= real_height; ++y) {
            corner.row = static_cast<size_t>(y - offset);

            for (int x = offset; x <= real_width; ++x) {
                int idx = board.idx[y][x];
                if (idx >= 0) {
                    corner.col = static_cast<size_t>(x - offset);
                    corner.major_pt = corners.pixels[idx];
                    corners_vec.push_back(corner);
                } else {
                    corners_vec.emplace_back(std::nullopt);
                    full_detected = false;
                }
            }
        }

        if (real_width < real_height) {
            int m = real_height;
            int n = real_width;
            std::vector<std::optional<ChessboardCorner>> new_corners(n * m);

            for (int old_row = 0; old_row < m; ++old_row) {
                for (int old_col = 0; old_col < n; ++old_col) {
                    int old_idx = old_row * n + old_col;
                    const auto &opt = corners_vec[old_idx];

                    corner.row = n - 1 - old_col;
                    corner.col = old_row;
                    int new_idx = corner.row * m + corner.col;

                    if (opt.has_value()) {
                        corner.major_pt = opt->major_pt;
                        new_corners[new_idx] = corner;
                    } else {
                        new_corners[new_idx] = std::nullopt;
                    }
                }
            }

            corners_vec = std::move(new_corners);
            std::swap(real_width, real_height);
        }

        // display = img.clone();
        // std::vector<cv::Point2d> gg;
        // for (const auto &pixel : pixels) {
        //     if (pixel.has_value()) {
        //         gg.push_back(pixel.value());
        //     }
        // }
        // plot_corners(display, gg, "ggg");
        chessboards.emplace_back(
            std::move(corners_vec),
            PatternSize(static_cast<size_t>(real_width), static_cast<size_t>(real_height)),
            full_detected
        );
    }
    cv::destroyAllWindows();

    return chessboards;
}


void ChessboardDetector::merge_raw_corners(
    ChessboardCorners &corners, ChessboardCorners &new_corners, double scale
) const {
    if (new_corners.pixels.empty()) {
        return;
    }

    std::for_each(new_corners.pixels.begin(), new_corners.pixels.end(), [&scale](auto &p) {
        p.x /= scale;
        p.y /= scale;
    });

    double merge_threshold = (scale > 1.0) ? 3.0 : 5.0;

    int added = 0;
    for (size_t i = 0; i < new_corners.pixels.size(); ++i) {
        const auto &new_point = new_corners.pixels[i];

        bool is_duplicate = false;
        for (const auto &existing_point : corners.pixels) {
            double dist = cv::norm(new_point - existing_point);
            if (dist < merge_threshold) {
                is_duplicate = true;
                break;
            }
        }

        if (!is_duplicate) {
            corners.pixels.push_back(new_corners.pixels[i]);
            corners.radius.push_back(new_corners.radius[i]);
            // corners.score.push_back(0.0);
            corners.edge_directions.push_back(new_corners.edge_directions[i]);
            added++;
        }
    }
}


void ChessboardDetector::raw_refinement(
    const cv::Mat &gray, const GradientSet &grad_set, ChessboardCorners &corners
) const {
    const cv::Mat &grad_x = grad_set.grad_x;
    const cv::Mat &grad_y = grad_set.grad_y;

    int width = gray.cols;
    int height = gray.rows;

    cv::parallel_for_(cv::Range(0, corners.pixels.size()), [&](const cv::Range &range) {
        for (int idx = range.start; idx < range.end; ++idx) {
            double corner_x = corners.pixels[idx].x;
            double corner_y = corners.pixels[idx].y;
            int patch_radius = corners.radius[idx];

            if (corner_x - patch_radius < 0 || corner_x + patch_radius >= width - 1 || corner_y - patch_radius < 0 ||
                corner_y + patch_radius >= height - 1) {
                continue;
            }

            cv::Mat hessian = cv::Mat::zeros(2, 2, CV_64F);
            cv::Mat rhs = cv::Mat::zeros(2, 1, CV_64F);

            cv::Mat grad_x_patch, grad_y_patch;
            grad_x_patch = extract_subpixel_patch(grad_x, cv::Point2d(corner_x, corner_y), patch_radius);
            grad_y_patch = extract_subpixel_patch(grad_y, cv::Point2d(corner_x, corner_y), patch_radius);

            int patch_size = 2 * patch_radius + 1;
            for (int row = 0; row < patch_size; ++row) {
                for (int col = 0; col < patch_size; ++col) {
                    double grad_x = grad_x_patch.at<double>(row, col);
                    double grad_y = grad_y_patch.at<double>(row, col);
                    double grad_magnitude = std::sqrt(grad_x * grad_x + grad_y * grad_y);

                    if (grad_magnitude < 0.1) {
                        continue;
                    }

                    if (col == patch_radius && row == patch_radius) {
                        continue;
                    }

                    double offset_x = col - patch_radius;
                    double offset_y = row - patch_radius;

                    hessian.at<double>(0, 0) += grad_x * grad_x;
                    hessian.at<double>(0, 1) += grad_x * grad_y;
                    hessian.at<double>(1, 0) += grad_x * grad_y;
                    hessian.at<double>(1, 1) += grad_y * grad_y;

                    rhs.at<double>(0, 0) +=
                        grad_x * grad_x * (corner_x + offset_x) + grad_x * grad_y * (corner_y + offset_y);
                    rhs.at<double>(1, 0) +=
                        grad_x * grad_y * (corner_x + offset_x) + grad_y * grad_y * (corner_y + offset_y);
                }
            }

            cv::Mat new_position = hessian.inv() * rhs;
            double new_x = new_position.at<double>(0, 0);
            double new_y = new_position.at<double>(1, 0);

            if (std::isnan(new_x) || std::isinf(new_x) || std::isnan(new_y) || std::isinf(new_y)) {
                continue; // не обновляем координаты
            }

            double displacement = std::abs(new_x - corner_x) + std::abs(new_y - corner_y);
            if (displacement < patch_radius * 2) {
                corners.pixels[idx].x = new_x;
                corners.pixels[idx].y = new_y;
            }
        }
    });
}


void ChessboardDetector::normalize(cv::Mat &img) const {
    if (params.norm) {
        cv::Mat blur_img;
        cv::blur(img, blur_img, cv::Size(params.norm_kernel_size, params.norm_kernel_size));

        double expected_min = -0.2;
        double expected_max = 0.2;
        double range = expected_max - expected_min;

        img = img - expected_min;
        img = cv::max(cv::min(img, range), 0.0);
        img = img / (expected_max - expected_min);
    } else {
        cv::normalize(img, img, 0.0, 1.0, cv::NORM_MINMAX);
    }
}


ChessboardDetector::ChessboardCorners ChessboardDetector::get_initial_corners(const cv::Mat &gray) const {
    ChessboardCorners corners;

    cv::Mat corner_response = cv::Mat::zeros(gray.size(), CV_64F);

    cv::Mat response_q1, response_q2, response_q3, response_q4;
    cv::Mat response_mean;

    cv::Mat white_score, black_score;
    cv::Mat score_case1, score_case2;

    int current_radius = -1;

    for (const auto &templates : templates_vec) {
        if (current_radius != templates.template_radius) {
            if (current_radius != -1) {
                auto new_corners = non_maximum_suppression(corner_response, 1, params.nms_threshold, current_radius);
                corners.pixels.insert(corners.pixels.end(), new_corners.begin(), new_corners.end());
                corners.radius.insert(corners.radius.end(), new_corners.size(), current_radius);
                // corners.score.insert(corners.score.end(), new_corners.size(), 0.0);
            }

            cv::Mat response_img;
            cv::normalize(corner_response, response_img, 0, 255, cv::NORM_MINMAX, CV_8U);
            current_radius = templates.template_radius;
            corner_response = cv::Mat::zeros(gray.size(), CV_64F);
        }

        cv::filter2D(gray, response_q1, -1, templates.top_right, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
        cv::filter2D(gray, response_q2, -1, templates.top_left, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
        cv::filter2D(gray, response_q4, -1, templates.bottom_right, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
        cv::filter2D(gray, response_q3, -1, templates.bottom_left, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);

        response_mean = (response_q1 + response_q2 + response_q3 + response_q4) / 4;

        // case 1: quadrants 1&3 are white, quadrants 2&4 are black
        white_score = cv::min(response_q1, response_q3) - response_mean;
        black_score = response_mean - cv::max(response_q2, response_q4);
        score_case1 = cv::min(white_score, black_score);

        // case 2: quadrants 1&3 are black, quadrants 2&4 are white
        white_score = response_mean - cv::max(response_q1, response_q3);
        black_score = cv::min(response_q2, response_q4) - response_mean;
        score_case2 = cv::min(white_score, black_score);

        corner_response = cv::max(corner_response, cv::max(score_case1, score_case2));
    }

    if (current_radius != -1) {
        auto new_corners = non_maximum_suppression(corner_response, 1, params.nms_threshold, current_radius);
        corners.pixels.insert(corners.pixels.end(), new_corners.begin(), new_corners.end());
        corners.radius.insert(corners.radius.end(), new_corners.size(), current_radius);
        // corners.score.insert(corners.score.end(), new_corners.size(), 0.0);
    }

    return corners;
}


void ChessboardDetector::create_grad_masks() {
    const double step = 0.3;
    const double interval = 2 * step;
    for (const auto &r : params.radius) {
        const int size = r * 2 + 1;
        grad_masks[r] = cv::Mat::zeros(size, size, CV_64F);

        cv::Mat &mat = grad_masks[r];
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                double dist = std::sqrt((j - r) * (j - r) + (i - r) * (i - r));
                dist /= r;
                dist = std::clamp(dist, 1.0 - step, 1.0 + step);
                mat.at<double>(i, j) = (1.0 + step - dist) / interval;
            }
        }
    }
}


std::vector<std::pair<int, double>> ChessboardDetector::meanshift(const std::vector<double> &hist, double sigma) const {
    std::unordered_map<int, double> hash_table;
    std::vector<std::pair<int, double>> modes;

    int r = static_cast<int>(std::round(2 * sigma));
    std::vector<double> gauss_kernel(2 * r + 1, 0);
    for (int idx = 0; idx < 2 * r + 1; idx++) {
        gauss_kernel[idx] = std::exp(-0.5 * (idx - r) * (idx - r) / sigma / sigma) / std::sqrt(2 * M_PI) / sigma;
    }

    int hist_size = hist.size();
    std::vector<double> hist_smoothed(hist_size, 0);
    for (int i = 0; i < hist_size; i++) {
        for (int j = 0; j < 2 * r + 1; j++) {
            hist_smoothed[(i + r) % hist_size] += hist[(i + j) % hist_size] * gauss_kernel[j];
        }
    }

    auto max_hist_val = std::max_element(hist_smoothed.begin(), hist_smoothed.end());
    if (*max_hist_val < EPSILON) {
        return modes;
    }

    std::vector<uint8_t> visited(hist_size, 0);
    for (int start_bin = 0; start_bin < hist_size; start_bin++) {
        int current_bin = start_bin;
        if (!visited[current_bin]) {
            while (true) {
                visited[current_bin] = true;
                int right_bin = (current_bin + 1) % hist_size;
                int left_bin = (current_bin + hist_size - 1) % hist_size;

                double current_val = hist_smoothed[current_bin];
                double right_val = hist_smoothed[right_bin];
                double left_val = hist_smoothed[left_bin];

                if (right_val >= current_val && right_val >= left_val) {
                    current_bin = right_bin;
                } else if (left_val > current_val && left_val > right_val) {
                    current_bin = left_bin;
                } else {
                    break;
                }
            }
            hash_table[current_bin] = hist_smoothed[current_bin];
        }
    }

    for (const auto &elem : hash_table) {
        modes.emplace_back(elem);
    }
    std::sort(modes.begin(), modes.end(), [](const auto &elem_1, const auto &elem_2) -> bool {
        return elem_1.second > elem_2.second;
    });

    return modes;
};


ChessboardDetector::ChessboardCorners
ChessboardDetector::filter_corners(const cv::Mat &gray, const GradientSet &grad_set, ChessboardCorners &corners) const {
    const cv::Mat &grad_img = grad_set.grad_img;
    const cv::Mat &angle_img = grad_set.angle_img;

    int width = gray.cols, height = gray.rows;
    ChessboardCorners filtred_corners;
    std::vector<uint8_t> valid(corners.pixels.size(), false);

    std::vector<double> cos_v(filter_params.circle_size);
    std::vector<double> sin_v(filter_params.circle_size);
    for (int idx = 0; idx < filter_params.circle_size; ++idx) {
        cos_v[idx] = std::cos(idx * 2.0 * M_PI / (filter_params.circle_size - 1));
        sin_v[idx] = std::sin(idx * 2.0 * M_PI / (filter_params.circle_size - 1));
    }

    cv::parallel_for_(cv::Range(0, corners.pixels.size()), [&](const cv::Range &range) -> void {
        for (int idx = range.start; idx < range.end; ++idx) {
            int num_crossings = 0;
            int num_modes = 0;

            int center_u = std::round(corners.pixels[idx].x);
            int center_v = std::round(corners.pixels[idx].y);
            int r = corners.radius[idx];
            if (center_u - r < 0 || center_u + r >= width - 1 || center_v - r < 0 || center_v + r >= height - 1) {
                continue;
            }

            std::vector<double> circle(filter_params.circle_size);
            for (int jdx = 0; jdx < filter_params.circle_size; ++jdx) {
                int circle_u = static_cast<int>(std::round(center_u + 0.75 * r * cos_v[jdx]));
                int circle_v = static_cast<int>(std::round(center_v + 0.75 * r * sin_v[jdx]));
                circle_u = std::clamp(circle_u, 0, width - 1);
                circle_v = std::clamp(circle_v, 0, height - 1);
                circle[jdx] = gray.at<double>(circle_v, circle_u);
            }

            auto minmax = std::minmax_element(circle.begin(), circle.end());
            double min = *minmax.first, max = *minmax.second;
            for (int jdx = 0; jdx < filter_params.circle_size; ++jdx) {
                circle[jdx] = circle[jdx] - min - (max - min) / 2;
            }

            int fisrt_cross_index = 0;
            for (int j = 0; j < filter_params.circle_size; ++j) {
                if ((circle[j] > 0) ^ (circle[(j + 1) % filter_params.circle_size] > 0)) {
                    fisrt_cross_index = (j + 1) % filter_params.circle_size;
                    break;
                }
            }
            for (int j = fisrt_cross_index, count = 1; j < filter_params.circle_size + fisrt_cross_index;
                 ++j, ++count) {
                if ((circle[j % filter_params.circle_size] > 0) ^ (circle[(j + 1) % filter_params.circle_size] > 0)) {
                    if (count >= filter_params.cross_threshold) {
                        ++num_crossings;
                    }
                    count = 1;
                }
            }

            int top_left_u = std::max(center_u - r, 0);
            int top_left_v = std::max(center_v - r, 0);
            int bottom_right_u = std::min(center_u + r, width - 1);
            int bottom_right_v = std::min(center_v + r, height - 1);
            cv::Mat img_weight_sub = cv::Mat::zeros(2 * r + 1, 2 * r + 1, CV_64F);
            grad_img.rowRange(top_left_v, bottom_right_v + 1)
                .colRange(top_left_u, bottom_right_u + 1)
                .copyTo(img_weight_sub(
                    cv::Range(top_left_v - center_v + r, bottom_right_v - center_v + r + 1),
                    cv::Range(top_left_u - center_u + r, bottom_right_u - center_u + r + 1)
                ));
            img_weight_sub = img_weight_sub.mul(grad_masks[r]);

            double tmp_maxval = 0;
            cv::minMaxLoc(img_weight_sub, NULL, &tmp_maxval);
            img_weight_sub.forEach<double>([&tmp_maxval](double &val, const int *pos) -> void {
                val = val < 0.5 * tmp_maxval ? 0 : val;
            });

            std::vector<double> hist(filter_params.bin_count, 0);
            for (int j2 = top_left_v; j2 <= bottom_right_v; ++j2) {
                for (int i2 = top_left_u; i2 <= bottom_right_u; ++i2) {
                    int bin =
                        static_cast<int>(std::floor(angle_img.at<double>(j2, i2) / (M_PI / filter_params.bin_count))) %
                        filter_params.bin_count;
                    hist[bin] += img_weight_sub.at<double>(j2 - center_v + r, i2 - center_u + r);
                }
            }

            auto modes = meanshift(hist);
            if (modes.empty()) {
                continue;
            }
            for (const auto &mode : modes) {
                if (2 * mode.second > modes[0].second) {
                    ++num_modes;
                }
            }

            // static std::mutex mtx;
            if (num_crossings == filter_params.target_num_crossing && num_modes == filter_params.target_num_modes) {
                // std::lock_guard<std::mutex> lock(mtx);
                valid[idx] = true;
            }
        }
    });

    for (int idx = 0; idx < corners.pixels.size(); ++idx) {
        if (valid[idx] == true) {
            filtred_corners.pixels.emplace_back(cv::Point2d(corners.pixels[idx].x, corners.pixels[idx].y));
            filtred_corners.radius.emplace_back(corners.radius[idx]);
            // filtred_corners.score.push_back(0.0);
        }
    }

    return filtred_corners;
};


ChessboardDetector::CorrelationTemplates::CorrelationTemplates(size_t r, std::pair<double, double> angle_pair) {
    int width = r * 2 + 1;
    int height = r * 2 + 1;

    cv::Point2i center(r, r);

    top_left = cv::Mat::zeros(height, width, CV_64F);
    top_right = cv::Mat::zeros(height, width, CV_64F);
    bottom_right = cv::Mat::zeros(height, width, CV_64F);
    bottom_left = cv::Mat::zeros(height, width, CV_64F);
    template_radius = r;

    auto angle_1 = angle_pair.first;
    auto angle_2 = angle_pair.second;

    std::pair<double, double> normal_1(-std::sin(angle_1), std::cos(angle_1));
    std::pair<double, double> normal_2(-std::sin(angle_2), std::cos(angle_2));

    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            cv::Point2i curr_point(i - center.y, j - center.x);
            double dist = std::sqrt(curr_point.x * curr_point.x + curr_point.y * curr_point.y);

            double s1 = curr_point.x * normal_1.first + curr_point.y * normal_1.second;
            double s2 = curr_point.x * normal_2.first + curr_point.y * normal_2.second;

            // skip pixels lying on the lines between the board cells
            double dead_zone = 0.1;
            if (dist <= r) {
                if (s1 <= -dead_zone && s2 <= -dead_zone) {
                    bottom_left.at<double>(j, i) = 1;  // Q3 (-,-)
                } else if (s1 >= dead_zone && s2 >= dead_zone) {
                    top_right.at<double>(j, i) = 1;    // Q1 (+,+)
                } else if (s1 <= -dead_zone && s2 >= dead_zone) {
                    top_left.at<double>(j, i) = 1;     // Q2 (-,+)
                } else if (s1 >= dead_zone && s2 <= -dead_zone) {
                    bottom_right.at<double>(j, i) = 1; // Q4 (+,-)
                }
            }
        }
    }

    auto normalize = [](cv::Mat &m) {
        double sum = cv::sum(m)[0];
        if (sum > 1e-5) {
            m /= sum;
        }
    };

    normalize(top_left);
    normalize(top_right);
    normalize(bottom_left);
    normalize(bottom_right);
}


void ChessboardDetector::create_templates_vec() {
    if (templates_vec.size() != 0) {
        return;
    }

    // clang-format off
    std::vector<double> templates_angels;
    if (params.precision_accuracy == false) {
        templates_angels = {0, M_PI_2, M_PI_4, -M_PI_4};
    } else {
        templates_angels = {
             0,             M_PI_2,        M_PI_4,   -M_PI_4,
             0,             M_PI_4,        0,        -M_PI_4,
             M_PI_4,        M_PI_2,       -M_PI_4,    M_PI_2,
            -3 * M_PI / 8,  3 * M_PI / 8, -M_PI / 8,  M_PI / 8,
            -M_PI / 8,     -3 * M_PI / 8,  M_PI / 8,  3 * M_PI / 8
        };
    }
    // clang-format on

    for (const auto &r : params.radius) {
        int width = r * 2 + 1;
        int height = r * 2 + 1;

        for (size_t idx = 0; idx < templates_angels.size(); idx += 2) {
            std::pair<double, double> angels_pair(templates_angels[idx], templates_angels[idx + 1]);
            CorrelationTemplates templates(r, std::move(angels_pair));
            templates_vec.emplace_back(templates);
        }
    }
}


std::vector<cv::Point2d>
ChessboardDetector::non_maximum_suppression(const cv::Mat &gray_image, int n, double tau, int margin) const {
    if (n <= 0 || gray_image.empty()) {
        return {};
    }
    if (margin < 0) {
        margin = 0;
    }

    cv::Mat roi = gray_image(cv::Rect(margin, margin, gray_image.cols - 2 * margin, gray_image.rows - 2 * margin));

    int step = n + 1;
    int blocks_y = (roi.rows - n + step - 1) / step;
    int blocks_x = (roi.cols - n + step - 1) / step;
    int total_blocks = blocks_y * blocks_x;

    // Вектор для сбора результатов из всех потоков (с защитой)
    std::vector<cv::Point2d> all_results;
    std::mutex result_mutex;

    parallel_for_(cv::Range(0, total_blocks), [&](const cv::Range &range) {
        std::vector<cv::Point2d> local_points;

        for (int i = range.start; i < range.end; i++) {
            int block_y = i / blocks_x;
            int block_x = i % blocks_x;

            int start_y = block_y * step;
            int start_x = block_x * step;
            int end_y = std::min(start_y + n, roi.rows);
            int end_x = std::min(start_x + n, roi.cols);

            double max_brightness = -std::numeric_limits<double>::max();
            cv::Point2i max_coord(-1, -1);

            for (int y = start_y; y < end_y; y++) {
                for (int x = start_x; x < end_x; x++) {
                    double brightness = roi.at<double>(y, x);
                    if (brightness > max_brightness) {
                        max_brightness = brightness;
                        max_coord = cv::Point2i(x, y);
                    }
                }
            }

            if (max_brightness < tau) {
                continue;
            }

            // Расширенная проверка локального максимума
            int check_start_x = std::max(max_coord.x - n, 0);
            int check_start_y = std::max(max_coord.y - n, 0);
            int check_end_x = std::min(max_coord.x + n, roi.cols);
            int check_end_y = std::min(max_coord.y + n, roi.rows);

            bool is_max = true;
            for (int y = check_start_y; y < check_end_y && is_max; ++y) {
                for (int x = check_start_x; x < check_end_x; ++x) {
                    if (roi.at<double>(y, x) > max_brightness) {
                        is_max = false;
                        break;
                    }
                }
            }

            if (is_max) {
                local_points.emplace_back(max_coord.x + margin, max_coord.y + margin);
            }
        }

        if (!local_points.empty()) {
            std::lock_guard<std::mutex> lock(result_mutex);
            all_results.insert(all_results.end(), local_points.begin(), local_points.end());
        }
    });

    // Удаление возможных дубликатов (разные блоки могли вернуть одну точку)
    std::sort(all_results.begin(), all_results.end(), [](const cv::Point2d &a, const cv::Point2d &b) {
        return (a.x < b.x) || (a.x == b.x && a.y < b.y);
    });
    all_results.erase(
        std::unique(
            all_results.begin(),
            all_results.end(),
            [](const cv::Point2d &a, const cv::Point2d &b) {
                return std::abs(a.x - b.x) < 1e-6 && std::abs(a.y - b.y) < 1e-6;
            }
        ),
        all_results.end()
    );

    return all_results;
}


std::vector<cv::Point2d>
non_maximum_suppression(const std::vector<std::pair<cv::Point2d, double>> &points, const cv::Size &img_size, int n) {
    if (n <= 0 || points.empty()) {
        return {};
    }

    cv::Mat points_map = cv::Mat::zeros(img_size, CV_64FC3);
    std::vector<cv::Point2d> processed_points;
    for (const auto &point : points) {
        const auto &pixel_coords = point.first;
        const auto &pixel_brightness = point.second;

        int map_y = cvRound(pixel_coords.y);
        int map_x = cvRound(pixel_coords.x);

        if (map_x >= 0 && map_x < points_map.cols && map_y >= 0 && map_y < points_map.rows) {
            auto &point_on_map = points_map.at<cv::Vec3d>(map_y, map_x);

            if (point_on_map[0] < pixel_brightness) {
                point_on_map[0] = pixel_brightness;
                point_on_map[1] = pixel_coords.x;
                point_on_map[2] = pixel_coords.y;
            }
        }
    }

    std::vector<cv::Point2d> all_results;
    cv::Mat result_mat = cv::Mat::zeros(img_size, CV_8U);

    auto double_limit_min = -std::numeric_limits<double>::max();
    parallel_for_(
        cv::Range(0, img_size.height),
        [&points_map, &result_mat, &img_size, n, double_limit_min](const cv::Range &range) {
            for (int y = range.start; y < range.end; y++) {
                for (int x = 0; x < img_size.width; x++) {
                    double max_brightness = points_map.at<cv::Vec3d>(y, x)[0];
                    if (max_brightness > 0.0) {

                        int start_x = std::max(x - n, 0);
                        int start_y = std::max(y - n, 0);
                        int end_x = std::min(x + n, img_size.width);
                        int end_y = std::min(y + n, img_size.height);

                        auto check_roi = [&]() -> bool {
                            for (int y_roi = start_y; y_roi < end_y; y_roi++) {
                                for (int x_roi = start_x; x_roi < end_x; x_roi++) {
                                    if (x == x_roi && y == y_roi) {
                                        continue;
                                    }
                                    double brightness = points_map.at<cv::Vec3d>(y_roi, x_roi)[0];
                                    if (brightness > max_brightness) {
                                        return false;
                                    }
                                }
                            }
                            return true;
                        };

                        if (check_roi()) {
                            result_mat.at<uint8_t>(y, x) = 1;
                        }
                    }
                }
            }
        }
    );

    for (int y = 0; y < result_mat.rows; y++) {
        for (int x = 0; x < result_mat.cols; x++) {
            if (result_mat.at<uint8_t>(y, x) == 1) {
                auto real_coord_x = points_map.at<cv::Vec3d>(y, x)[1];
                auto real_coord_y = points_map.at<cv::Vec3d>(y, x)[2];
                all_results.emplace_back(cv::Point2d(real_coord_x, real_coord_y));
            }
        }
    }

    return all_results;
}


cv::Mat ChessboardDetector::extract_subpixel_patch(const cv::Mat &img, const cv::Point2d &center, int radius) const {
    int grid_x = static_cast<int>(center.x);
    int grid_y = static_cast<int>(center.y);

    double frac_x = center.x - grid_x;
    double frac_y = center.y - grid_y;

    double weight_top_left = (1 - frac_x) * (1 - frac_y);
    double weight_top_right = frac_x * (1 - frac_y);
    double weight_bottom_left = (1 - frac_x) * frac_y;
    double weight_bottom_right = frac_x * frac_y;

    cv::Mat patch(2 * radius + 1, 2 * radius + 1, CV_64F);

    for (int row_offset = -radius; row_offset <= radius; row_offset++) {
        for (int col_offset = -radius; col_offset <= radius; col_offset++) {
            int x_left = grid_x + col_offset;
            int x_right = x_left + 1;
            int y_top = grid_y + row_offset;
            int y_bottom = y_top + 1;

            patch.at<double>(row_offset + radius, col_offset + radius) =
                weight_top_left * img.at<double>(y_top, x_left) + weight_top_right * img.at<double>(y_top, x_right) +
                weight_bottom_left * img.at<double>(y_bottom, x_left) +
                weight_bottom_right * img.at<double>(y_bottom, x_right);
        }
    }

    return patch;
}


cv::Mat ChessboardDetector::extract_subpixel_patch(
    const cv::Mat &img, const cv::Point2d &center, const cv::Mat &mask, int radius
) const {
    int grid_x = static_cast<int>(center.x);
    int grid_y = static_cast<int>(center.y);

    double frac_x = center.x - grid_x;
    double frac_y = center.y - grid_y;

    double weight_top_left = (1 - frac_x) * (1 - frac_y);
    double weight_top_right = frac_x * (1 - frac_y);
    double weight_bottom_left = (1 - frac_x) * frac_y;
    double weight_bottom_right = frac_x * frac_y;

    auto rows = (2 * radius + 1) * (2 * radius + 1);
    cv::Mat patch(rows, 1, CV_64F);

    size_t count_non_zeros_elements = 0;
    for (int row_offset = -radius; row_offset <= radius; row_offset++) {
        for (int col_offset = -radius; col_offset <= radius; col_offset++) {
            int x_left = grid_x + col_offset;
            int x_right = x_left + 1;
            int y_top = grid_y + row_offset;
            int y_bottom = y_top + 1;

            if (mask.at<double>(row_offset + radius, col_offset + radius) < EPSILON) {
                continue;
            } else {
                // clang-format off
                patch.at<double>(count_non_zeros_elements, 0) = 
                    weight_top_left * img.at<double>(y_top, x_left) +
                    weight_top_right * img.at<double>(y_top, x_right) +
                    weight_bottom_left * img.at<double>(y_bottom, x_left) +
                    weight_bottom_right * img.at<double>(y_bottom, x_right);
                // clang-format on
                count_non_zeros_elements++;
            }
        }
    }

    patch.resize(count_non_zeros_elements);
    return patch;
}


std::pair<cv::Mat, size_t> ChessboardDetector::create_cone_filter(int r) const {
    cv::Mat kernel(2 * r + 1, 2 * r + 1, CV_64F);

    double sum = 0.0;

    size_t count_zero_elements = 0;
    for (int row_offset = -r; row_offset <= r; row_offset++) {
        for (int col_offset = -r; col_offset <= r; col_offset++) {
            kernel.at<double>(row_offset + r, col_offset + r) = std::max(
                0.0, static_cast<double>(r) + 1.0 - std::sqrt(row_offset * row_offset + col_offset * col_offset)
            );
            sum += kernel.at<double>(row_offset + r, col_offset + r);
            if (kernel.at<double>(row_offset + r, col_offset + r) < EPSILON) {
                count_zero_elements++;
            }
        }
    }

    kernel /= sum;

    return std::pair(std::move(kernel), count_zero_elements);
}


void ChessboardDetector::polynomial_fit(const cv::Mat &gray, ChessboardCorners &corners, int r) const {
    const int MAX_ITERATIONS = 5;
    const double EPS = 0.01;
    int width = gray.cols;
    int height = gray.rows;

    std::vector<uint8_t> valid(corners.pixels.size(), false);
    ChessboardCorners fit_corners;

    auto cone_filter = create_cone_filter(r);
    cv::Mat cone_kernel = cone_filter.first;
    size_t count_zeros_elements = cone_filter.second;

    cv::Mat blur;
    cv::filter2D(gray, blur, -1, cone_kernel, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);

    // f(x, y) = k0 * x^2 + k1 * y^2 + k2 * x * y + k3 * x + k4 * y + k5
    cv::Mat polynomial_design = cv::Mat::zeros((2 * r + 1) * (2 * r + 1) - count_zeros_elements, 6, CV_64F);
    int curr_row = 0;
    for (int row_offset = -r; row_offset <= r; row_offset++) {
        for (int col_offset = -r; col_offset <= r; col_offset++) {
            if (cone_kernel.at<double>(row_offset + r, col_offset + r) >= EPSILON) {
                polynomial_design.at<double>(curr_row, 0) = col_offset * col_offset;
                polynomial_design.at<double>(curr_row, 1) = row_offset * row_offset;
                polynomial_design.at<double>(curr_row, 2) = col_offset * row_offset;
                polynomial_design.at<double>(curr_row, 3) = col_offset;
                polynomial_design.at<double>(curr_row, 4) = row_offset;
                polynomial_design.at<double>(curr_row, 5) = 1;
                curr_row++;
            }
        }
    }
    cv::Mat least_squares_solver =
        (polynomial_design.t() * polynomial_design).inv(cv::DECOMP_SVD) * polynomial_design.t();

    cv::parallel_for_(cv::Range(0, corners.pixels.size()), [&](const cv::Range &range) -> void {
        for (int i = range.start; i < range.end; ++i) {
            double x = corners.pixels[i].x;
            double y = corners.pixels[i].y;
            double curr_x = x;
            double curr_y = y;
            bool is_saddle_point = true;

            for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
                if (curr_x - r < 0 || curr_x + r >= width - 1 || curr_y - r < 0 || curr_y + r >= height - 1) {
                    is_saddle_point = false;
                    break;
                }

                cv::Mat patch = extract_subpixel_patch(blur, cv::Point2d(curr_x, curr_y), cone_kernel, r);
                cv::Mat k = least_squares_solver * patch;

                double det = 4 * k.at<double>(0, 0) * k.at<double>(1, 0) - k.at<double>(2, 0) * k.at<double>(2, 0);
                if (det > 0) {
                    is_saddle_point = false;
                    break;
                }

                double dx =
                    (k.at<double>(2, 0) * k.at<double>(4, 0) - 2 * k.at<double>(1, 0) * k.at<double>(3, 0)) / det;
                double dy =
                    (k.at<double>(2, 0) * k.at<double>(3, 0) - 2 * k.at<double>(0, 0) * k.at<double>(4, 0)) / det;

                curr_x += dx;
                curr_y += dy;

                double dist = std::sqrt((curr_x - x) * (curr_x - x) + (curr_y - y) * (curr_y - y));
                if (dist > r) {
                    is_saddle_point = false;
                    break;
                }
                if (std::sqrt(dx * dx + dy * dy) <= EPS) {
                    break;
                }
            }

            if (is_saddle_point) {
                valid[i] = true;
                corners.pixels[i] = cv::Point2d(curr_x, curr_y);
            }
        }
    });

    for (int idx = 0; idx < corners.pixels.size(); idx++) {
        if (valid[idx] == true) {
            fit_corners.edge_directions.emplace_back(corners.edge_directions[idx]);
            fit_corners.radius.emplace_back(corners.radius[idx]);
            fit_corners.pixels.emplace_back(corners.pixels[idx]);
        }
    }
    corners = std::move(fit_corners);
}


std::vector<std::array<double, 2>> ChessboardDetector::get_corner_orientation(GradientSet &grad_set) const {
    cv::Mat &grad_img = grad_set.grad_img;
    cv::Mat &angle_img = grad_set.angle_img;

    angle_img.forEach<double>([](double &val, const int *pos) -> void {
        val += M_PI / 2;
        val = val >= M_PI ? val - M_PI : val;
    });

    std::vector<double> hist(filter_params.bin_count, 0);
    for (int i = 0; i < angle_img.cols; ++i) {
        for (int j = 0; j < angle_img.rows; ++j) {
            int bin = static_cast<int>(std::floor(angle_img.at<double>(j, i) / (M_PI / filter_params.bin_count))) %
                      filter_params.bin_count;
            hist[bin] += grad_img.at<double>(j, i);
        }
    }

    auto modes = meanshift(hist, 1.5);
    if (modes.size() < 2) {
        return std::vector<std::array<double, 2>>();
    }

    double angle_first = modes[0].first * M_PI / filter_params.bin_count + M_PI / filter_params.bin_count / 2;
    double angle_second = modes[1].first * M_PI / filter_params.bin_count + M_PI / filter_params.bin_count / 2;
    if (angle_first > angle_second) {
        std::swap(angle_first, angle_second);
    }

    double delta_angle = std::min(angle_second - angle_first, angle_first + M_PI - angle_second);
    // 0.314 rad == 18 degrees == 180 * 0.1
    if (delta_angle <= 0.314) {
        return std::vector<std::array<double, 2>>();
    }

    std::vector<std::array<double, 2>> corner_orientation(2);
    corner_orientation[0] = {std::cos(angle_first), std::sin(angle_first)};
    corner_orientation[1] = {std::cos(angle_second), std::sin(angle_second)};

    // std::sort(corner_orientation.begin(), corner_orientation.end(), [](const auto &a1, const auto &a2) {
    //     return a1[0] * a2[1] - a1[1] * a2[0] > 0;
    // });

    // corner_orientation.back()[0] = -corner_orientation.back()[0];
    // corner_orientation.back()[1] = -corner_orientation.back()[1];

    // std::sort(corner_orientation.begin(), corner_orientation.end(), [](const auto &a1, const auto &a2) {
    //     return a1[0] * a2[1] - a1[1] * a2[0] > 0;
    // });

    return corner_orientation;
}


void ChessboardDetector::refine_corners(GradientSet &grad_set, ChessboardCorners &corners) const {
    const cv::Mat &grad_x = grad_set.grad_x;
    const cv::Mat &grad_y = grad_set.grad_y;
    const cv::Mat &angle_img = grad_set.angle_img;
    const cv::Mat &grad_imd = grad_set.grad_img;

    ChessboardCorners refinement_corners;

    int width = grad_x.cols, height = grad_x.rows;
    std::vector<cv::Point2d> corners_out_p, corners_out_v1, corners_out_v2;
    std::vector<size_t> corners_out_r;
    std::vector<uint8_t> valid(corners.pixels.size(), false);

    refinement_corners.edge_directions.resize(corners.pixels.size());

    cv::parallel_for_(cv::Range(0, corners.pixels.size()), [&](const cv::Range &range) -> void {
        for (int i = range.start; i < range.end; ++i) {
            int grid_x = std::round(corners.pixels[i].x);
            int grid_y = std::round(corners.pixels[i].y);
            double x = corners.pixels[i].x;
            double y = corners.pixels[i].y;
            size_t r = corners.radius[i];

            if (grid_x - r < 0 || grid_x + r >= width - 1 || grid_y - r < 0 || grid_y + r >= height - 1) {
                continue;
            }

            cv::Mat img_angle_patch = extract_subpixel_patch(angle_img, cv::Point2d(grid_x, grid_y), r);
            cv::Mat img_weight_patch = extract_subpixel_patch(grad_imd, cv::Point2d(grid_x, grid_y), r);
            img_weight_patch = img_weight_patch.mul(grad_masks[r]);

            GradientSet curr_set;
            curr_set.angle_img = std::move(img_angle_patch);
            curr_set.grad_img = std::move(img_weight_patch);

            auto corner_orientation = get_corner_orientation(curr_set);
            if (corner_orientation.empty()) {
                continue;
            }

            cv::Mat A_first = cv::Mat::zeros(2, 2, CV_64F);
            cv::Mat A_second = cv::Mat::zeros(2, 2, CV_64F);
            for (int j2 = grid_y - r; j2 <= grid_y + r; ++j2) {
                for (int i2 = grid_x - r; i2 <= grid_x + r; ++i2) {
                    double dx = grad_x.at<double>(j2, i2);
                    double dy = grad_y.at<double>(j2, i2);
                    double grad = std::sqrt(dx * dx + dy * dy);
                    if (grad < 0.1) {
                        continue;
                    }

                    double dx_norm = dx / grad;
                    double dy_norm = dy / grad;

                    auto refinement_orientation = [&](const std::array<double, 2> &dir, cv::Mat &A) {
                        if (std::abs(dx_norm * dir[0] + dy_norm * dir[1]) < 0.25) {
                            A.at<double>(0, 0) += dx * dx;
                            A.at<double>(0, 1) += dx * dy;
                            A.at<double>(1, 0) += dx * dy;
                            A.at<double>(1, 1) += dy * dy;
                        }
                    };

                    refinement_orientation(corner_orientation[0], A_first);
                    refinement_orientation(corner_orientation[1], A_second);
                }
            }

            auto update_orientation = [&](const cv::Mat &A, std::array<double, 2> &dir) {
                cv::Mat eigenvalues, eigenvectors;
                cv::eigen(A, eigenvalues, eigenvectors);
                dir[0] = eigenvectors.at<double>(1, 0);
                dir[1] = eigenvectors.at<double>(1, 1);
            };

            update_orientation(A_first, corner_orientation[0]);
            update_orientation(A_second, corner_orientation[1]);

            auto sort_condition = [](const auto &elem1, const auto &elem2) {
                return elem1[0] * elem2[1] - elem1[1] * elem2[0] > 0;
            };

            std::sort(corner_orientation.begin(), corner_orientation.end(), sort_condition);
            corner_orientation[corner_orientation.size() - 1][0] =
                -corner_orientation[corner_orientation.size() - 1][0];
            corner_orientation[corner_orientation.size() - 1][1] =
                -corner_orientation[corner_orientation.size() - 1][1];
            std::sort(corner_orientation.begin(), corner_orientation.end(), sort_condition);

            if (params.polynomial_fit) {
                valid[i] = true;
            } else {
                auto new_corner = full_refinement_corner(cv::Point2d(x, y), grad_set, corner_orientation, r);
                if (new_corner.has_value()) {
                    valid[i] = true;
                    corners.pixels[i] = new_corner.value();
                }
            }

            refinement_corners.edge_directions[i] = {
                {cv::Point2d(corner_orientation[0][0], corner_orientation[0][1]),
                 cv::Point2d(corner_orientation[1][0], corner_orientation[1][1])}
            };
        }
    });

    auto raw_orientations = std::move(refinement_corners.edge_directions);
    for (int idx = 0; idx < corners.pixels.size(); ++idx) {
        if (valid[idx] == true) {
            refinement_corners.pixels.emplace_back(corners.pixels[idx]);
            refinement_corners.radius.emplace_back(corners.radius[idx]);
            refinement_corners.edge_directions.emplace_back(raw_orientations[idx]);
        }
    }
    corners = std::move(refinement_corners);
}


std::optional<cv::Point2d> ChessboardDetector::full_refinement_corner(
    const cv::Point2d &p, GradientSet &grad_set, std::vector<std::array<double, 2>> &corner_orientation, double r
) const {
    const cv::Mat &grad_x = grad_set.grad_x;
    const cv::Mat &grad_y = grad_set.grad_y;

    const int MAX_ITERATION = 5;
    const double EPS = 0.01;
    const double MAX_DIST = 3;

    double curr_x = p.x;
    double curr_y = p.y;
    double last_x = p.x;
    double last_y = p.y;

    for (int iteration = 0; iteration < MAX_ITERATION; ++iteration) {
        cv::Mat hessian = cv::Mat::zeros(2, 2, CV_64F);
        cv::Mat rhs = cv::Mat::zeros(2, 1, CV_64F);

        cv::Mat grad_x_patch, grad_y_patch;
        if (curr_x - r < 0 || curr_x + r >= grad_x.cols || curr_y - r < 0 || curr_y + r >= grad_x.rows) {
            break;
        }
        grad_x_patch = extract_subpixel_patch(grad_x, cv::Point2d(curr_x, curr_y), r);
        grad_y_patch = extract_subpixel_patch(grad_y, cv::Point2d(curr_x, curr_y), r);

        for (int row = 0; row < 2 * r + 1; ++row) {
            for (int col = 0; col < 2 * r + 1; ++col) {
                double dx = grad_x_patch.at<double>(row, col);
                double dy = grad_y_patch.at<double>(row, col);
                double grad = std::sqrt(dx * dx + dy * dy);
                if (grad < 0.1) {
                    continue;
                }

                double dx_norm = dx / grad;
                double dy_norm = dy / grad;

                if (col == r && row == r) {
                    continue;
                }

                double offset_x = col - r;
                double offset_y = row - r;

                auto distance_to_direction = [&offset_x, &offset_y](const std::array<double, 2> &dir) -> double {
                    double projection = offset_x * dir[0] + offset_y * dir[1];

                    double dist_x = offset_x - projection * dir[0];
                    double dist_y = offset_y - projection * dir[1];

                    return std::sqrt(dist_x * dist_x + dist_y * dist_y);
                };

                auto gradient_aligned_with = [&dx_norm, &dy_norm](const std::array<double, 2> &dir) -> bool {
                    return std::abs(dx_norm * dir[0] + dy_norm * dir[1]) < 0.25;
                };

                double dist_to_dir1 = distance_to_direction(corner_orientation[0]);
                double dist_to_dir2 = distance_to_direction(corner_orientation[1]);

                bool corresponds_to_dir1 = (dist_to_dir1 < MAX_DIST) && gradient_aligned_with(corner_orientation[0]);
                bool corresponds_to_dir2 = (dist_to_dir2 < MAX_DIST) && gradient_aligned_with(corner_orientation[1]);

                if (corresponds_to_dir1 || corresponds_to_dir2) {
                    hessian.at<double>(0, 0) += dx * dx;
                    hessian.at<double>(0, 1) += dx * dy;
                    hessian.at<double>(1, 0) += dx * dy;
                    hessian.at<double>(1, 1) += dy * dy;

                    double pixel_x = curr_x + offset_x;
                    double pixel_y = curr_y + offset_y;

                    rhs.at<double>(0, 0) += dx * dx * pixel_x + dx * dy * pixel_y;
                    rhs.at<double>(1, 0) += dx * dy * pixel_x + dy * dy * pixel_y;
                }
            }
        }

        cv::Mat new_position = hessian.inv() * rhs;
        last_x = curr_x;
        last_y = curr_y;
        curr_x = new_position.at<double>(0, 0);
        curr_y = new_position.at<double>(1, 0);
        double dist = std::sqrt((curr_x - last_x) * (curr_x - last_x) + (curr_y - last_y) * (curr_y - last_y));
        if (dist >= MAX_DIST) {
            curr_x = last_x;
            curr_y = last_y;
            break;
        }
        if (dist <= EPS) {
            break;
        }
    }

    if (std::sqrt((curr_x - p.x) * (curr_x - p.x) + (curr_y - p.y) * (curr_y - p.y)) < std::max(r / 2, MAX_DIST)) {
        return cv::Point2d(curr_x, curr_y);
    } else {
        return std::nullopt;
    }
}


double ChessboardDetector::get_corner_score(
    const cv::Mat &patch, const cv::Mat &grad_patch, std::array<cv::Point2d, 2> &edge_direction
) const {
    double center = static_cast<double>((patch.cols - 1)) / 2;
    cv::Mat filter = cv::Mat::ones(patch.size(), CV_64F) * -1;
    for (int i = 0; i < patch.cols; ++i) {
        for (int j = 0; j < patch.rows; ++j) {
            cv::Point2d p1{i - center, j - center};
            cv::Point2d p2{
                (p1.x * edge_direction[0].x + p1.y * edge_direction[0].y) * edge_direction[0].x,
                (p1.x * edge_direction[0].x + p1.y * edge_direction[0].y) * edge_direction[0].y
            };
            cv::Point2d p3{
                (p1.x * edge_direction[1].x + p1.y * edge_direction[1].y) * edge_direction[1].x,
                (p1.x * edge_direction[1].x + p1.y * edge_direction[1].y) * edge_direction[1].y
            };
            if (cv::norm(p1 - p2) <= 1.5 || cv::norm(p1 - p3) <= 1.5) {
                filter.at<double>(j, i) = 1;
            }
        }
    }

    // double min_val, max_val;
    // cv::minMaxLoc(patch, &min_val, &max_val);
    // std::cout << "Patch values: min=" << min_val << " max=" << max_val << " range=" << max_val - min_val <<
    // std::endl;

    cv::Scalar mean, std;
    cv::meanStdDev(filter, mean, std);
    filter = (filter - mean[0]) / std[0];
    cv::meanStdDev(grad_patch, mean, std);
    cv::Mat grad_patch_norm = (grad_patch - mean[0]) / std[0];

    double score_gradient = cv::sum(grad_patch_norm.mul(filter))[0];
    score_gradient = std::max(score_gradient / (patch.cols * patch.rows - 1), 0.);

    // double a1 = std::atan2(edge_direction[0].y, edge_direction[0].x) * 180 / M_PI;
    // double a2 = std::atan2(edge_direction[1].y, edge_direction[1].x) * 180 / M_PI;
    // std::cerr << "Angles: " << a1 << "°, " << a2 << "°" << std::endl;
    // std::cerr << "Difference: " << std::abs(a1 - a2) << "°" << std::endl;

    std::pair<double, double> angle_pair(
        std::atan2(edge_direction[0].y, edge_direction[0].x), std::atan2(edge_direction[1].y, edge_direction[1].x)
    );

    size_t r = (patch.cols - 1) / 2;
    CorrelationTemplates correlation_template(r, std::move(angle_pair));

    double q1 = cv::sum(patch.mul(correlation_template.top_right))[0];
    double q2 = cv::sum(patch.mul(correlation_template.top_left))[0];
    double q3 = cv::sum(patch.mul(correlation_template.bottom_left))[0];
    double q4 = cv::sum(patch.mul(correlation_template.bottom_right))[0];

    double response_mean = (q1 + q2 + q3 + q4) / 4;

    // case 1: quadrants 1&3 are white, quadrants 2&4 are black
    double white_score = cv::min(q1, q3) - response_mean;
    double black_score = response_mean - cv::min(q2, q4);
    double score_case1 = cv::min(white_score, black_score);

    // case 2: quadrants 1&3 are black, quadrants 2&4 are white
    white_score = response_mean - cv::min(q1, q3);
    black_score = cv::min(q2, q4) - response_mean;
    double score_case2 = cv::min(white_score, black_score);

    double score_intensity = std::max(std::max(score_case1, score_case2), 0.);

    // std::mutex m;
    // std::lock_guard<std::mutex> lock(m);
    // std::cerr << "Corner 0: G=" << score_gradient << " I=" << score_intensity << std::endl;

    // std::cerr << "q1=" << q1 << " q2=" << q2 << " q3=" << q3 << " q4=" << q4 << " mean=" << response_mean <<
    // std::endl;
    return score_gradient * score_intensity;
}


void ChessboardDetector::calculate_corners_score(
    const cv::Mat &gray, GradientSet &grad_set, ChessboardCorners &corners
) const {
    cv::Mat &grad_img = grad_set.grad_img;
    int width = gray.cols;
    int height = gray.rows;

    ChessboardCorners final_corners;
    std::vector<uint8_t> valid(corners.pixels.size(), false);
    corners.score.resize(corners.pixels.size());

    cv::parallel_for_(cv::Range(0, corners.pixels.size()), [&](const cv::Range &range) -> void {
        for (int idx = range.start; idx < range.end; ++idx) {
            double u = corners.pixels[idx].x;
            double v = corners.pixels[idx].y;
            int r = corners.radius[idx];

            if (u - r < 0 || u + r >= width - 1 || v - r < 0 || v + r >= height - 1) {
                corners.score[idx] = 0.;
                continue;
            }

            cv::Mat gray_patch = extract_subpixel_patch(gray, cv::Point2d(u, v), r);
            cv::Mat grad_img_patch = extract_subpixel_patch(grad_img, cv::Point2d(u, v), r);
            grad_img_patch = grad_img_patch.mul(grad_masks[r]);

            double score = get_corner_score(gray_patch, grad_img_patch, corners.edge_directions[idx]);
            if (score > params.score_threshold) {
                valid[idx] = true;
                corners.score[idx] = score;
            } else {
                corners.score[idx] = 0.0;
            }
            // valid[idx] = true;
        }
    });

    for (size_t idx = 0; idx < corners.pixels.size(); idx++) {
        if (valid[idx] == true) {
            final_corners.pixels.emplace_back(corners.pixels[idx]);
            final_corners.radius.emplace_back(corners.radius[idx]);
            final_corners.edge_directions.emplace_back(corners.edge_directions[idx]);
            final_corners.score.emplace_back(corners.score[idx]);
        }
    }

    corners = std::move(final_corners);
}


int ChessboardDetector::find_neighbor_along_dir(
    const ChessboardCorners &corners, const std::vector<uint8_t> &used, int idx, const cv::Point2d &target_dir
) const {
    std::vector<double> dists(corners.pixels.size(), DOUBLE_MAX);

    for (int jdx = 0; jdx < corners.pixels.size(); ++jdx) {
        if (used[jdx]) {
            continue;
        }
        cv::Point2d curr_dir = corners.pixels[jdx] - corners.pixels[idx];
        double dist_along = curr_dir.dot(target_dir);
        double dist_cross = cv::norm(curr_dir - dist_along * target_dir);
        double dist = dist_along + 5 * dist_cross;
        if (dist_along >= 0) {
            dists[jdx] = dist;
        }
    }

    int neighbor_idx = std::min_element(dists.begin(), dists.end()) - dists.begin();
    double min_dist = dists[neighbor_idx];
    if (std::abs(min_dist - DOUBLE_MAX) < 1) {
        return -1;
    }
    return neighbor_idx;
}


std::optional<ChessboardDetector::RawBoard>
ChessboardDetector::init_board(const ChessboardCorners &corners, std::vector<uint8_t> &used, int idx) const {
    if (corners.pixels.size() < 9) {
        return std::nullopt;
    }

    RawBoard board;
    board.idx.assign(3, std::vector<int>(3, -1));

    auto dir_first = corners.edge_first(idx);
    auto dir_second = corners.edge_second(idx);
    board.idx[1][1] = idx;
    used[idx] = true;

    auto clear_bad_board = [&board, &used]() {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                int corner_idx = board.idx[i][j];
                if (corner_idx >= 0 && corner_idx < used.size()) {
                    used[corner_idx] = false;
                }
                board.idx[i][j] = -1;
            }
        }
    };

    auto set_neighbor = [&](int &cell, const cv::Point2d &dir) -> bool {
        int neighbor = find_neighbor_along_dir(corners, used, idx, dir);
        if (neighbor == -1) {
            return false;
        }
        cell = neighbor;
        used[neighbor] = true;
        return true;
    };

    // clang-format off
    if (!set_neighbor(board.idx[1][0], -dir_first) ||
        !set_neighbor(board.idx[1][2], dir_first) ||
        !set_neighbor(board.idx[0][1], -dir_second) ||
        !set_neighbor(board.idx[2][1], dir_second)) 
    {
        clear_bad_board();
        return std::nullopt;
    }
    // clang-format on

    auto get_diagonal_neighbor =
        [&corners,
         &used,
         this](int idx_first, int idx_second, const cv::Point2d &dir_first, const cv::Point2d &dir_second) -> int {
        int new_idx_first;
        int new_idx_second;
        double dist_first;
        double dist_second;

        new_idx_first = find_neighbor_along_dir(corners, used, idx_first, dir_first);
        new_idx_second = find_neighbor_along_dir(corners, used, idx_second, dir_second);

        if (new_idx_first != new_idx_second) {
            if (new_idx_first >= 0) {
                dist_first = std::abs(
                    cv::norm(corners.pixels[new_idx_first] - corners.pixels[idx_first]) -
                    cv::norm(corners.pixels[new_idx_first] - corners.pixels[idx_second])
                );
            } else {
                dist_first = DOUBLE_MAX;
            }
            if (new_idx_second >= 0) {
                dist_second = std::abs(
                    cv::norm(corners.pixels[new_idx_second] - corners.pixels[idx_first]) -
                    cv::norm(corners.pixels[new_idx_second] - corners.pixels[idx_second])
                );
            } else {
                dist_second = DOUBLE_MAX;
            }
            if (dist_first > dist_second) {
                std::swap(new_idx_first, new_idx_second);
            }
        }
        if (new_idx_first >= 0) {
            used[new_idx_first] = true;
        }
        return new_idx_first;
    };

    board.idx[0][0] = get_diagonal_neighbor(board.idx[1][0], board.idx[0][1], -dir_second, -dir_first);
    board.idx[0][2] = get_diagonal_neighbor(board.idx[1][2], board.idx[0][1], -dir_second, dir_first);
    board.idx[2][0] = get_diagonal_neighbor(board.idx[1][0], board.idx[2][1], dir_second, -dir_first);
    board.idx[2][2] = get_diagonal_neighbor(board.idx[1][2], board.idx[2][1], dir_second, dir_first);

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (board.idx[i][j] == -1) {
                clear_bad_board();
                return std::nullopt;
            }
        }
    }

    board.energy = std::move(
        std::vector<std::vector<std::vector<double>>>(
            3, std::vector<std::vector<double>>(3, std::vector<double>(3, DOUBLE_MAX))
        )
    );

    return board;
}


cv::Point3i ChessboardDetector::board_energy(const ChessboardCorners &corners, RawBoard &board) const {
    double E_corners = -1.0 * board.num_corners();

    double max_E_struct = std::numeric_limits<double>::min();
    int res_x = 0, res_y = 0, res_z = 0;

    auto process_triple = [&](int i, int j, int idx_first, int idx_second, int idx_third, int z) {
        if (idx_first >= 0 && idx_second >= 0 && idx_third >= 0) {
            const cv::Point2d &x1 = corners.pixels[idx_first];
            const cv::Point2d &x2 = corners.pixels[idx_second];
            const cv::Point2d &x3 = corners.pixels[idx_third];

            double E_structure = cv::norm(x1 + x3 - 2 * x2) / cv::norm(x1 - x3);
            board.energy[i][j][z] = E_corners * (1 - E_structure);

            if (E_structure > max_E_struct) {
                max_E_struct = E_structure;
                res_x = j;
                res_y = i;
                res_z = z;
            }
        }
    };

    // horizontal walk
    for (int i = 0; i < board.idx.size(); ++i) {
        for (int j = 0; j < board.idx[i].size() - 2; ++j) {
            process_triple(i, j, board.idx[i][j], board.idx[i][j + 1], board.idx[i][j + 2], 0);
        }
    }

    // vertical walk
    for (int i = 0; i < board.idx.size() - 2; ++i) {
        for (int j = 0; j < board.idx[i].size(); ++j) {
            process_triple(i, j, board.idx[i][j], board.idx[i + 1][j], board.idx[i + 2][j], 1);
        }
    }

    return {res_x, res_y, res_z};
}


double ChessboardDetector::find_minE(const RawBoard &board, const cv::Point2i &p) const {
    struct Check {
        int dx, dy, dir;
    };
    // clang-format off
    std::vector<Check> checks = {
        { 0, 0, 0}, { 0,  0, 1}, {0,  0, 2},
        {-1, 0, 0}, {-1, -1, 1}, {0, -1, 2},
        {-2, 0, 0}, {-2, -2, 1}, {0, -2, 2}
    };
    // clang-format on

    double minE = std::numeric_limits<double>::max();
    for (const auto &c : checks) {
        int nx = p.x + c.dx, ny = p.y + c.dy;
        if (nx >= 0 && ny >= 0 && ny < board.energy.size() && nx < board.energy[ny].size()) {
            minE = std::min(minE, board.energy[ny][nx][c.dir]);
        }
    }
    return minE;
}


void ChessboardDetector::filter_board(
    const ChessboardCorners &corners,
    std::vector<uint8_t> &used,
    RawBoard &board,
    std::vector<cv::Point2i> &proposal,
    double &energy
) const {
    while (!proposal.empty()) {
        cv::Point3i maxE_pos = board_energy(corners, board);
        double p_energy = board.energy[maxE_pos.y][maxE_pos.x][maxE_pos.z];
        // std::cout << std::setprecision(15) << "energy " << energy << std::endl;
        // std::cout << std::setprecision(15) << "p_energy " << p_energy << std::endl;
        // std::cout << "=== FILTER BOARD ===" << std::endl;
        // std::cout << "proposal.size() = " << proposal.size() << std::endl;
        // std::cout << "maxE_pos = (" << maxE_pos.x << "," << maxE_pos.y << "," << maxE_pos.z << ")" << std::endl;
        if (p_energy <= energy) {
            energy = p_energy;
            break;
        }
        if (!params.handle_occlusions) {
            for (const auto &p : proposal) {
                used[board.idx[p.y][p.x]] = false;
                board.idx[p.y][p.x] = -2;
            }
            return;
        }

        cv::Point2i triple[3] = {
            {maxE_pos.x, maxE_pos.y},
            (maxE_pos.z == 0) ? cv::Point2i{maxE_pos.x + 1, maxE_pos.y} : cv::Point2i{maxE_pos.x, maxE_pos.y + 1},
            (maxE_pos.z == 0) ? cv::Point2i{maxE_pos.x + 2, maxE_pos.y} : cv::Point2i{maxE_pos.x, maxE_pos.y + 2}
        };

        double corner_quality[3];
        for (int i = 0; i < 3; ++i) {
            corner_quality[i] = find_minE(board, triple[i]);
        }

        double max_quality = -std::numeric_limits<double>::max();
        int worst_idx = 0;

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < proposal.size(); ++j) {
                if (proposal[j].x == triple[i].x && proposal[j].y == triple[i].y && corner_quality[i] > max_quality) {
                    max_quality = corner_quality[i];
                    worst_idx = j;
                    maxE_pos.x = proposal[j].x;
                    maxE_pos.y = proposal[j].y;
                }
            }
        }

        if (max_quality == -std::numeric_limits<double>::max()) {
            break;
        }

        // std::cout << "worst_idx = " << worst_idx << std::endl;
        // std::cout << "removing corner at board grid (" << maxE_pos.y << "," << maxE_pos.x << ")" << std::endl;
        // std::cout << "global corner index = " << board.idx[maxE_pos.y][maxE_pos.x] << std::endl;

        proposal.erase(proposal.begin() + worst_idx);
        used[board.idx[maxE_pos.y][maxE_pos.x]] = false;
        board.idx[maxE_pos.y][maxE_pos.x] = -2;
    }
}


std::vector<cv::Point2d> ChessboardDetector::predict_corners(
    const ChessboardCorners &corners,
    const std::vector<int> &far_indices,
    const std::vector<int> &middle_indices,
    const std::vector<int> &near_indices
) const {
    constexpr double CONSERVATIVE_PREDICTION_FACTOR = 0.75;

    std::vector<cv::Point2d> predicted_positions(near_indices.size());

    if (far_indices.empty()) {
        for (int i = 0; i < predicted_positions.size(); ++i) {
            predicted_positions[i] = 2 * corners.pixels[near_indices[i]] - corners.pixels[middle_indices[i]];
        }
    } else {
        for (int i = 0; i < predicted_positions.size(); ++i) {
            cv::Point2d vec_far_to_middle = corners.pixels[middle_indices[i]] - corners.pixels[far_indices[i]];
            cv::Point2d vec_middle_to_near = corners.pixels[near_indices[i]] - corners.pixels[middle_indices[i]];

            double angle_far_to_middle = std::atan2(vec_far_to_middle.y, vec_far_to_middle.x);
            double angle_middle_to_near = std::atan2(vec_middle_to_near.y, vec_middle_to_near.x);
            double predicted_angle = 2 * angle_middle_to_near - angle_far_to_middle;

            double scale_far_to_middle = cv::norm(vec_far_to_middle);
            double scale_middle_to_near = cv::norm(vec_middle_to_near);
            double predicted_scale = 2 * scale_middle_to_near - scale_far_to_middle;

            predicted_positions[i].x = corners.pixels[near_indices[i]].x +
                                       CONSERVATIVE_PREDICTION_FACTOR * predicted_scale * std::cos(predicted_angle);
            predicted_positions[i].y = corners.pixels[near_indices[i]].y +
                                       CONSERVATIVE_PREDICTION_FACTOR * predicted_scale * std::sin(predicted_angle);
        }
    }

    return predicted_positions;
}


std::vector<int> ChessboardDetector::predict_board_corners(
    const ChessboardCorners &corners,
    std::vector<uint8_t> &used,
    std::vector<int> &p1,
    std::vector<int> &p2,
    std::vector<int> &p3
) const {
    std::vector<cv::Point2d> predicted_positions = predict_corners(corners, p1, p2, p3);
    std::vector<int> matched_corner_indices(predicted_positions.size(), -2);

    std::vector<std::vector<double>> distance_mat(
        predicted_positions.size(), std::vector<double>(corners.pixels.size(), DOUBLE_MAX)
    );

    for (int pred_idx = 0; pred_idx < predicted_positions.size(); ++pred_idx) {
        cv::Point2d vec_to_pred = predicted_positions[pred_idx] - corners.pixels[p3[pred_idx]];

        for (int corner_idx = 0; corner_idx < corners.pixels.size(); ++corner_idx) {
            if (used[corner_idx]) {
                continue;
            }

            cv::Point2d vec_to_candidate = corners.pixels[corner_idx] - corners.pixels[p3[pred_idx]];

            cv::Point2d projected_coords(
                vec_to_candidate.dot(vec_to_pred), vec_to_candidate.dot(cv::Point2d(vec_to_pred.y, -vec_to_pred.x))
            );

            projected_coords = projected_coords / (cv::norm(vec_to_pred) * cv::norm(vec_to_pred));

            double angular_deviation = std::atan2(projected_coords.y, projected_coords.x);
            double distance_deviation = (1 - cv::norm(projected_coords));

            distance_mat[pred_idx][corner_idx] = std::sqrt(
                std::abs(angular_deviation) +
                distance_deviation * distance_deviation * distance_deviation * distance_deviation
            );
        }
    }

    for (int iteration = 0; iteration < predicted_positions.size(); ++iteration) {
        double best_distance = DOUBLE_MAX;
        int best_prediction_idx = 0;
        int best_corner_idx = 0;

        for (int pred_idx = 0; pred_idx < predicted_positions.size(); ++pred_idx) {
            int best_corner_for_this_pred =
                std::min_element(distance_mat[pred_idx].begin(), distance_mat[pred_idx].end()) -
                distance_mat[pred_idx].begin();

            double distance = distance_mat[pred_idx][best_corner_for_this_pred];

            if (distance < best_distance) {
                best_distance = distance;
                best_corner_idx = best_corner_for_this_pred;
                best_prediction_idx = pred_idx;
            }
        }

        if (DOUBLE_MAX - best_distance < EPSILON) {
            break;
        }

        for (auto &value : distance_mat[best_prediction_idx]) {
            value = DOUBLE_MAX;
        }
        for (int pred_idx = 0; pred_idx < predicted_positions.size(); ++pred_idx) {
            distance_mat[pred_idx][best_corner_idx] = DOUBLE_MAX;
        }

        matched_corner_indices[best_prediction_idx] = best_corner_idx;
        used[best_corner_idx] = true;
    }

    return matched_corner_indices;
}


bool ChessboardDetector::add_board_bound(RawBoard &board, int direction) const {
    int rows = board.idx.size(), cols = board.idx[0].size();

    auto need_to_grow = [&](auto &&has_corner) {
        bool need_grow = false;
        for (int i = 0; i < (direction % 2 ? rows : cols); ++i) {
            if (has_corner(i)) {
                need_grow = true;
                break;
            }
        }
        return need_grow;
    };

    auto grow_row = [&](auto &&add_row) {
        std::vector<int> new_row(cols, -1);
        std::vector<std::vector<double>> new_energy(cols, std::vector<double>(3, DOUBLE_MAX));
        add_row(new_row, new_energy);
    };

    auto grow_col = [&](auto &&add_col) {
        for (int i = 0; i < rows; ++i) {
            add_col(i);
        }
    };

    switch (direction) {
    case 0: // up
        if (need_to_grow([&](int i) {
                return board.idx[0][i] != -2 && board.idx[0][i] != -1;
            })) {
            grow_row([&](std::vector<int> &new_row, std::vector<std::vector<double>> &new_energy) {
                board.idx.insert(board.idx.begin(), new_row);
                board.energy.insert(board.energy.begin(), new_energy);
            });
            return true;
        }
        break;

    case 1: // left
        if (need_to_grow([&](int i) {
                return board.idx[i][0] != -2 && board.idx[i][0] != -1;
            })) {
            grow_col([&](int row) {
                board.idx[row].insert(board.idx[row].begin(), -1);
                board.energy[row].insert(board.energy[row].begin(), std::vector<double>(3, DOUBLE_MAX));
            });
            return true;
        }
        break;

    case 2: // down
        if (need_to_grow([&](int i) {
                return board.idx[rows - 1][i] != -2 && board.idx[rows - 1][i] != -1;
            })) {
            grow_row([&](std::vector<int> &new_row, std::vector<std::vector<double>> &new_energy) {
                board.idx.emplace_back(new_row);
                board.energy.emplace_back(new_energy);
            });
            return true;
        }
        break;

    case 3: // right
        if (need_to_grow([&](int i) {
                return board.idx[i][cols - 1] != -2 && board.idx[i][cols - 1] != -1;
            })) {
            grow_col([&](int row) {
                board.idx[row].emplace_back(-1);
                board.energy[row].emplace_back(std::vector<double>(3, DOUBLE_MAX));
            });
            return true;
        }
        break;
    }

    return false;
}


void ChessboardDetector::handle_occlusions(
    const ChessboardCorners &corners,
    RawBoard &board,
    std::vector<cv::Point2i> &proposal,
    std::vector<uint8_t> &used,
    int direction
) const {
    int cols = board.idx[0].size();
    int rows = board.idx.size();
    std::vector<int> idx, p1, p2, p3, pred;

    if (params.handle_occlusions) {
        const int offsets[4][6] = {
            // up idx1(i+3,j), idx2(i+2,j), idx3(i+1,j)
            {3, 0, 2, 0, 1, 0},
            // left idx1(i,j+3), idx2(i,j+2), idx3(i,j+1)
            {0, 3, 0, 2, 0, 1},
            // down idx1(i-3,j), idx2(i-2,j), idx3(i-1,j)
            {-3, 0, -2, 0, -1, 0},
            // right idx1(i,j-3), idx2(i,j-2), idx3(i,j-1)
            {0, -3, 0, -2, 0, -1}
        };

        const int bounds[4][6] = {
            {rows - 4, -1, -1, 0, cols, 1}, // up
            {0, rows, 1, cols - 4, -1, -1}, // left
            {3, rows, 1, 0, cols, 1},       // down
            {0, rows, 1, 3, cols, 1}        // right
        };

        const auto &b = bounds[direction];
        const auto &o = offsets[direction];

        for (int i = b[0]; i != b[1]; i += b[2]) {
            for (int j = b[3]; j != b[4]; j += b[5]) {
                int idx1 = board.idx[i + o[0]][j + o[1]];
                int idx2 = board.idx[i + o[2]][j + o[3]];
                int idx3 = board.idx[i + o[4]][j + o[5]];
                if (board.idx[i][j] == -1 && idx1 >= 0 && idx2 >= 0 && idx3 >= 0) {
                    p1.emplace_back(idx1);
                    p2.emplace_back(idx2);
                    p3.emplace_back(idx3);
                    proposal.emplace_back(j, i);
                }
            }
        }

        if (proposal.empty() && !params.strict_board_grow) {
            int i_start = b[0] + (b[2] > 0 ? 1 : -1);
            int j_start = b[3] + (b[5] > 0 ? 1 : -1);

            for (int i = i_start; i != b[1]; i += b[2]) {
                for (int j = j_start; j != b[4]; j += b[5]) {
                    int idx2 = board.idx[i + o[2]][j + o[3]];
                    int idx3 = board.idx[i + o[4]][j + o[5]];
                    if (board.idx[i][j] == -1 && idx2 >= 0 && idx3 >= 0) {
                        p2.emplace_back(idx2);
                        p3.emplace_back(idx3);
                        proposal.emplace_back(j, i);
                    }
                }
            }
        }

        pred = predict_board_corners(corners, used, p1, p2, p3);
        for (int i = 0; i < proposal.size(); ++i) {
            board.idx[proposal[i].y][proposal[i].x] = pred[i];
        }
    }
}


bool ChessboardDetector::grow_board_dir(
    const ChessboardCorners &corners,
    RawBoard &board,
    std::vector<cv::Point2i> &proposal,
    std::vector<uint8_t> &used,
    int direction
) const {
    int cols = board.idx[0].size();
    int rows = board.idx.size();
    std::vector<int> idx, p1, p2, p3, pred;

    const int boundary_offsets[4][6] = {
        // up idx1(3,i), idx2(2,i), idx3(1,i)
        {3, 0, 2, 0, 1, 0},
        // left idx1(i,3), idx2(i,2), idx3(i,1)
        {0, 3, 0, 2, 0, 1},
        // down idx1(rows-4,i), idx2(rows-3,i), idx3(rows-2,i)
        {-4, 0, -3, 0, -2, 0},
        // up idx1(i,cols-4), idx2(i,cols-3), idx3(i,cols-2)
        {0, -4, 0, -3, 0, -2}
    };

    const int boundary_loops[4][4] = {
        {0, 0, cols, 1},        // up    proposal(i,0)
        {1, 0, rows, 1},        // left  proposal(0,i)
        {0, rows - 1, cols, 1}, // down  proposal(i,rows-1)
        {1, cols - 1, rows, 1}  // right proposal(cols-1,i)
    };

    const auto &o = boundary_offsets[direction];
    const auto &l = boundary_loops[direction];

    if (l[0] == 0) {
        for (int i = 0; i < l[2]; i += l[3]) {
            int idx1 = board.idx[o[0] < 0 ? rows + o[0] : o[0]][i + o[1]];
            int idx2 = board.idx[o[2] < 0 ? rows + o[2] : o[2]][i + o[3]];
            int idx3 = board.idx[o[4] < 0 ? rows + o[4] : o[4]][i + o[5]];

            if (idx1 >= 0 && idx2 >= 0 && idx3 >= 0) {
                p1.emplace_back(idx1);
                p2.emplace_back(idx2);
                p3.emplace_back(idx3);
                proposal.emplace_back(i, l[1]);
            }
        }
    } else {
        for (int i = 0; i < l[2]; i += l[3]) {
            int idx1 = board.idx[i + o[0]][o[1] < 0 ? cols + o[1] : o[1]];
            int idx2 = board.idx[i + o[2]][o[3] < 0 ? cols + o[3] : o[3]];
            int idx3 = board.idx[i + o[4]][o[5] < 0 ? cols + o[5] : o[5]];

            if (idx1 >= 0 && idx2 >= 0 && idx3 >= 0) {
                p1.emplace_back(idx1);
                p2.emplace_back(idx2);
                p3.emplace_back(idx3);
                proposal.emplace_back(l[1], i);
            }
        }
    }

    pred = predict_board_corners(corners, used, p1, p2, p3);

    if (!params.handle_occlusions) {
        for (int i = 0; i < proposal.size(); ++i) {
            if (pred[i] < 0) {
                proposal.clear();
                return false;
            }
        }
    }

    // bool any_success = false;
    // for (int i = 0; i < pred.size(); ++i) {
    //     if (pred[i] >= 0) {
    //         any_success = true;
    //         break;
    //     }
    // }
    // if (!any_success) {
    //     proposal.clear();
    //     return false;
    // }

    for (int i = 0; i < proposal.size(); ++i) {
        board.idx[proposal[i].y][proposal[i].x] = pred[i];
    }

    return true;
}


ChessboardDetector::GrowStatus ChessboardDetector::grow_board(
    const ChessboardCorners &corners,
    std::vector<uint8_t> &used,
    RawBoard &board,
    std::vector<cv::Point2i> &proposal,
    int direction
) const {
    if (board.idx.empty()) {
        return GrowStatus::FAILED;
    }

    handle_occlusions(corners, board, proposal, used, direction);

    if (!proposal.empty()) {
        return GrowStatus::INTERNAL;
    }

    if (!add_board_bound(board, direction)) {
        return GrowStatus::FAILED;
    }

    if (!grow_board_dir(corners, board, proposal, used, direction)) {
        return GrowStatus::FAILED;
    }

    if (proposal.empty()) {
        return GrowStatus::FAILED;
    }

    return GrowStatus::EXTERNAL;
}


std::vector<ChessboardDetector::RawBoard>
ChessboardDetector::boards_from_corners(const cv::Mat &img, const ChessboardCorners &corners) const {
    std::vector<RawBoard> boards;
    std::vector<uint8_t> used(corners.pixels.size(), false);

    int start = 0;
    if (!params.overlay) {
        static thread_local std::random_device rd;
        static thread_local std::mt19937 gen(rd());
        std::uniform_int_distribution<> distrib(0, corners.pixels.size() - 1);
        start = distrib(gen);
    }

    int n = 0;
    while (n++ < corners.pixels.size()) {
        int seed_idx = (n + start) % corners.pixels.size();

        if (used[seed_idx]) {
            continue;
        }

        auto board_opt = init_board(corners, used, seed_idx);
        if (!board_opt.has_value()) {
            continue;
        }

        RawBoard board = board_opt.value();

        cv::Point3i maxE_pos = board_energy(corners, board);
        double energy = board.energy[maxE_pos.y][maxE_pos.x][maxE_pos.z];

        if (energy > -6.0) {
            for (int jj = 0; jj < 3; ++jj) {
                for (int ii = 0; ii < 3; ++ii) {
                    used[board.idx[jj][ii]] = false;
                }
            }
            continue;
        }

        // std::vector<cv::Point2d> corners_viz;
        // for (const auto &idx : board.idx) {
        //     for (const auto &jdx : idx) {
        //         if (jdx >= 0) {
        //             corners_viz.push_back(corners.pixels.at(jdx));
        //         }
        //     }
        // }
        // cv::Mat debug_img;
        // img.convertTo(debug_img, CV_8U, 255.0);
        // plot_corners(debug_img, corners_viz, "init");
        // cv::cvtColor(img, viz, cv::COLOR_GRAY2BGR);
        bool grew;
        do {
            grew = false;
            int num_corners_before = board.num_corners();

            for (int direction = 0; direction < 4; ++direction) {
                std::vector<cv::Point2i> proposal;
                GrowStatus grow_status = grow_board(corners, used, board, proposal, direction);

                if (grow_status == GrowStatus::FAILED) {
                    // std::cout << "G" << std::endl;
                    continue;
                }
                // std::cout << "grow_status=" << static_cast<int>(grow_status) << " proposal.size=" <<
                // proposal.size()
                //           << std::endl;

                filter_board(corners, used, board, proposal, energy);

                // std::cout << "grow_status after filter=" << static_cast<int>(grow_status)
                //           << " proposal.size=" << proposal.size() << std::endl;

                // bool added = false;
                // for (const auto &p : proposal) {
                //     if (board.idx[p.y][p.x] >= 0) {
                //         added = true;
                //     }
                // }

                if (grow_status == GrowStatus::INTERNAL) {
                    // if (added) {
                    --direction;
                    // }
                }

                grew = true;

                // std::vector<cv::Point2d> corners_viz2;
                // for (const auto &idx : board.idx) {
                //     for (const auto &jdx : idx) {
                //         if (jdx >= 0) {
                //             corners_viz2.push_back(corners.pixels.at(jdx));
                //         }
                //     }
                // }
                // cv::Mat debug_img;
                // img.convertTo(debug_img, CV_8U, 255.0);
                // plot_corners(debug_img, corners_viz2, "iter");
            }

            if (board.num_corners() == num_corners_before) {
                break;
            }
        } while (grew);

        if (!params.overlay) {
            boards.emplace_back(board);
            continue;
        }

        std::vector<std::pair<int, double>> overlaps;

        for (int board_idx = 0; board_idx < boards.size(); ++board_idx) {
            const auto &existing_board = boards[board_idx];
            bool has_overlap = false;

            for (int r1 = 0; r1 < board.idx.size() && !has_overlap; ++r1) {
                for (int c1 = 0; c1 < board.idx[0].size() && !has_overlap; ++c1) {
                    int corner_idx = board.idx[r1][c1];

                    if (corner_idx < 0) {
                        continue;
                    }

                    for (int r2 = 0; r2 < existing_board.idx.size() && !has_overlap; ++r2) {
                        for (int c2 = 0; c2 < existing_board.idx[0].size() && !has_overlap; ++c2) {
                            if (corner_idx == existing_board.idx[r2][c2]) {
                                has_overlap = true;
                                cv::Point3i maxE_pos_tmp = board_energy(corners, boards[board_idx]);
                                overlaps.emplace_back(
                                    board_idx, boards[board_idx].energy[maxE_pos_tmp.y][maxE_pos_tmp.x][maxE_pos_tmp.z]
                                );
                            }
                        }
                    }
                }
            }
        }

        if (overlaps.empty()) {
            boards.push_back(board);
        } else {
            bool is_better = true;
            for (const auto &[idx, e] : overlaps) {
                if (e <= energy) {
                    is_better = false;
                    break;
                }
            }

            if (is_better) {
                std::vector<uint8_t> keep(boards.size(), true);
                for (const auto &[idx, _] : overlaps) {
                    keep[idx] = false;
                }

                std::vector<RawBoard> filtered_boards;
                for (int idx = 0; idx < boards.size(); ++idx) {
                    if (keep[idx]) {
                        filtered_boards.push_back(boards[idx]);
                    }
                }

                boards = std::move(filtered_boards);
                boards.push_back(board);
            }
        }

        std::fill(used.begin(), used.end(), false);
        n += 2;
    }

    return boards;
}
}; // namespace ccl
