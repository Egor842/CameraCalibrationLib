#include "../include/LineDetector.hpp"


namespace ccl {


constexpr double EPSILON = 1e-6;
constexpr double DOUBLE_MAX = std::numeric_limits<double>::max();
constexpr double ANGLE_TOLERANCE = (22.5 / 180) * M_PI;
constexpr double NFA_PROBABILITY = 0.125;


enum class LinesCombineCase
{
    START_START = 0,
    START_END,
    END_START,
    END_END
};


Line::Line(double A, double B, double C, cv::Point2d start, cv::Point2d end, size_t len, size_t first_segment_pixel_idx)
    : A(A),
      B(B),
      C(C),
      start(start),
      end(end),
      len(len),
      first_segment_pixel_idx(first_segment_pixel_idx) {};


double Line::get_A() const noexcept {
    return A;
}


double Line::get_B() const noexcept {
    return B;
}


double Line::get_C() const noexcept {
    return C;
}


cv::Point2d Line::get_start() const noexcept {
    return start;
}


cv::Point2d Line::get_end() const noexcept {
    return end;
}


size_t Line::get_len() const noexcept {
    return len;
}


size_t Line::get_first_segment_pixel() const noexcept {
    return first_segment_pixel_idx;
}


void Line::set_A(double new_A) noexcept {
    A = new_A;
}


void Line::set_B(double new_B) noexcept {
    B = new_B;
}


void Line::set_C(double new_C) noexcept {
    C = new_C;
}


void Line::set_start(cv::Point2d new_start) noexcept {
    start = new_start;
}


void Line::set_end(cv::Point2d new_end) noexcept {
    end = new_end;
}


void Line::set_len(size_t new_len) noexcept {
    len = new_len;
}


void Line::set_first_segment_pixel(size_t new_idx) noexcept {
    first_segment_pixel_idx = new_idx;
}


void Line::update_params(
    cv::Point2d new_start = cv::Point2d(-1, -1), cv::Point2d new_end = cv::Point2d(-1, -1)
) noexcept {
    if (new_start.x != -1 && new_start.y != -1) {
        start = new_start;
    }
    if (new_end.x != -1 && new_end.y != -1) {
        end = new_end;
    }
    A = start.y - end.y;
    B = end.x - start.x;
    C = start.x * end.y - end.x * start.y;
}


std::vector<Line> LineDetector::detect(const cv::Mat &img) const {
    std::vector<Line> lines;

    cv::Mat gray;
    if (img.channels() == 3) {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else if (img.channels() == 1) {
        gray = img;
    }

    double logNT = 2.0 * (log10(static_cast<double>(gray.cols)) + log10(static_cast<double>(gray.rows)));
    nfa = std::make_unique<NFA>(NFA_PROBABILITY, logNT);

    auto raw_detection = std::move(edge_detector->detect(gray));
    const auto &segments = raw_detection.second;

    for (const auto &segment : segments) {
        auto new_lines = split_segment(segment);
        if (new_lines.empty()) {
            continue;
        }
        new_lines = combine_collinear_lines(segment, new_lines);
        for (const auto &line : new_lines) {
            bool valid = validate_line_on_segment(gray, line, segment);
            if (valid == true) {
                lines.push_back(line);
            }
        }
    }

    cv::Mat display;
    if (img.channels() == 1) {
        cv::cvtColor(img, display, cv::COLOR_GRAY2BGR);
    } else {
        display = img.clone();
    }

    for (const auto &line : lines) {
        cv::line(display, line.get_start(), line.get_end(), cv::Scalar(0, 255, 0), 1, cv::LINE_8, 0);
        cv::imshow("Detected Lines", display);
        cv::waitKey(100);
    }

    cv::imshow("Detected Lines", display);
    cv::imwrite("Detected_Lines.png", display);
    cv::waitKey(0);

    return lines;
}


std::vector<Line> LineDetector::raw_detect(const cv::Mat &img) const {
    std::vector<Line> lines;

    cv::Mat gray;
    if (img.channels() == 3) {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else if (img.channels() == 1) {
        gray = img;
    }

    double logNT = 2.0 * (log10(static_cast<double>(gray.cols)) + log10(static_cast<double>(gray.rows)));
    nfa = std::make_unique<NFA>(NFA_PROBABILITY, logNT);

    auto raw_detection = std::move(edge_detector->detect(gray));
    const auto &segments = raw_detection.second;

    for (const auto &segment : segments) {
        auto new_lines = split_segment(segment);
        if (new_lines.empty()) {
            continue;
        }
        new_lines = combine_collinear_lines(segment, new_lines);
        lines.insert(lines.end(), new_lines.begin(), new_lines.end());
    }

    cv::Mat display;
    if (img.channels() == 1) {
        cv::cvtColor(img, display, cv::COLOR_GRAY2BGR);
    } else {
        display = img.clone();
    }

    for (const auto &line : lines) {
        cv::line(display, line.get_start(), line.get_end(), cv::Scalar(0, 255, 0), 1, cv::LINE_8, 0);
        cv::imshow("Detected Lines", display);
        cv::waitKey(100);
    }

    cv::imshow("Detected Lines", display);
    cv::imwrite("Detected_Lines.png", display);
    cv::waitKey(0);

    return lines;
}


double LineDetector::compute_distance_between_lines(
    const Line &line1, const Line &line2, std::function<bool(double current_dist, double target_dist)> comparator
) const noexcept {
    double dist = cv::norm(line1.get_start() - line2.get_start());
    double target_dist = dist;

    dist = cv::norm(line1.get_start() - line2.get_end());
    if (comparator(dist, target_dist)) {
        target_dist = dist;
    }

    dist = cv::norm(line1.get_end() - line2.get_start());
    if (comparator(dist, target_dist)) {
        target_dist = dist;
    }

    dist = cv::norm(line1.get_end() - line2.get_end());
    if (comparator(dist, target_dist)) {
        target_dist = dist;
    }

    return target_dist;
}


std::vector<Line> LineDetector::combine_collinear_lines(const Segment &segment, const std::vector<Line> &lines) const {
    std::vector<Line> res;

    auto try_combine = [this](Line &combine_res, const Line &insertable_line) -> bool {
        double dist = compute_distance_between_lines(combine_res, insertable_line, [](double dist, double target_dist) {
            return target_dist < dist;
        });
        if (dist > params.max_distance_between_lines) {
            return false;
        }

        const Line *line_a = &insertable_line;
        const Line *line_b = &combine_res;

        if (cv::norm(line_a->get_start() - line_a->get_end()) > cv::norm(line_b->get_start() - line_b->get_end())) {
            line_a = &combine_res;
            line_b = &insertable_line;
        }

        dist = std::abs(
            line_b->get_A() * line_a->get_start().x + line_b->get_B() * line_a->get_start().y + line_b->get_C()
        );
        dist += std::abs(
            line_b->get_A() * (line_a->get_start().x + line_a->get_end().x) / 2.0 +
            line_b->get_B() * (line_a->get_start().y + line_a->get_end().y) / 2.0 + line_b->get_C()
        );
        dist +=
            std::abs(line_b->get_A() * line_a->get_end().x + line_b->get_B() * line_a->get_end().y + line_b->get_C());
        if ((dist / 3.0) > params.max_error) {
            return false;
        }

        /// case 1: (s1, s2)
        cv::Point2d dist_point(combine_res.get_start() - insertable_line.get_start());
        dist = std::abs(dist_point.x) + std::abs(dist_point.y);
        double max = dist;
        LinesCombineCase combine_case = LinesCombineCase::START_START;

        /// case 2: (s1, e2)
        dist_point = cv::Point2d(combine_res.get_start() - insertable_line.get_end());
        dist = std::abs(dist_point.x) + std::abs(dist_point.y);
        if (dist > max) {
            max = dist;
            combine_case = LinesCombineCase::START_END;
        }

        /// case 3: (e1, s2)
        dist_point = cv::Point2d(combine_res.get_end() - insertable_line.get_start());
        dist = std::abs(dist_point.x) + std::abs(dist_point.y);
        if (dist > max) {
            max = dist;
            combine_case = LinesCombineCase::END_START;
        }

        /// case 4: (e1, e2)
        dist_point = cv::Point2d(combine_res.get_end() - insertable_line.get_end());
        dist = std::abs(dist_point.x) + std::abs(dist_point.y);
        if (dist > max) {
            max = dist;
            combine_case = LinesCombineCase::END_END;
        }

        switch (combine_case) {
        case LinesCombineCase::START_START:
            combine_res.set_end(insertable_line.get_start());
            break;
        case LinesCombineCase::START_END:
            combine_res.set_end(insertable_line.get_end());
            break;
        case LinesCombineCase::END_START:
            combine_res.set_start(insertable_line.get_start());
            break;
        case LinesCombineCase::END_END:
            combine_res.set_start(combine_res.get_end());
            combine_res.set_end(insertable_line.get_end());
            break;
        default:
            break;
        }

        if (combine_res.get_first_segment_pixel() + combine_res.get_len() + 5 >=
            insertable_line.get_first_segment_pixel()) {
            auto len = combine_res.get_len() + insertable_line.get_len();
            combine_res.set_len(len);
        } else if (insertable_line.get_len() > combine_res.get_len()) {
            combine_res.set_first_segment_pixel(insertable_line.get_first_segment_pixel());
            combine_res.set_len(insertable_line.get_len());
        }

        combine_res.update_params();

        return true;
    };

    if (lines.empty()) {
        return res;
    }

    Line curr_line = lines[0];
    for (size_t idx = 1; idx < lines.size(); idx++) {
        if (try_combine(curr_line, lines[idx]) == false) {
            res.push_back(std::move(curr_line));
            curr_line = lines[idx];
        }
    }
    res.push_back(curr_line);

    return res;
}


std::vector<Line> LineDetector::split_segment(const Segment &segment) const noexcept {
    size_t it_start = 0;
    size_t it_end = params.min_line_len - 1;
    LineCoeffs line_coeffs;
    std::vector<Line> lines;

    auto calculate_dist = [&line_coeffs, &segment](size_t index) -> double {
        return std::abs(line_coeffs[0] * segment[index].x + line_coeffs[1] * segment[index].y + line_coeffs[2]);
    };

    while (segment.size() - it_start >= params.min_line_len) {
        bool valid = false;

        while (segment.size() - it_start >= params.min_line_len) {
            line_coeffs = std::move(line_fit(segment, it_start, it_end));
            if (line_coeffs[3] <= 0.5) {
                valid = true;
                break;
            }

            it_start += 1;
            it_end += 1;
        }

        if (valid == false) {
            return std::vector<Line>();
        }

        size_t index = it_start + params.min_line_len;
        size_t len = params.min_line_len;

        while (index < segment.size()) {
            size_t last_good_index = index - 1;
            size_t good_pixel_count = 0;
            size_t bad_pixel_count = 0;

            while (index < segment.size()) {
                double dist = calculate_dist(index);

                if (dist <= params.line_error) {
                    last_good_index = index;
                    good_pixel_count++;
                    bad_pixel_count = 0;
                } else {
                    bad_pixel_count++;
                    if (bad_pixel_count >= 5) {
                        break;
                    }
                }

                index++;
            }

            if (good_pixel_count >= 2) {
                line_coeffs = std::move(line_fit(segment, it_start, last_good_index));
                index = last_good_index + 1;
            }

            if (good_pixel_count < 2 || index >= segment.size()) {
                cv::Point2d start_line;
                cv::Point2d end_line;

                int idx = it_start;
                while (calculate_dist(idx) > params.line_error) {
                    idx++;
                }
                start_line = projection_point_to_line(line_coeffs, segment[idx]);
                size_t skipped_pixels = idx - it_start;

                idx = last_good_index;
                while (calculate_dist(idx) > params.line_error) {
                    idx--;
                }
                end_line = projection_point_to_line(line_coeffs, segment[idx]);

                if (cv::norm(end_line - start_line) <= EPSILON) {
                    break;
                }
                lines.emplace_back(
                    line_coeffs[0],
                    line_coeffs[1],
                    line_coeffs[2],
                    start_line,
                    end_line,
                    idx - it_start - skipped_pixels + 1,
                    it_start + skipped_pixels
                );
                len = idx - it_start - skipped_pixels + 1;
                break;
            }
        }

        it_start += len;
        it_end = it_start + params.min_line_len - 1;
    }

    return lines;
}


LineDetector::LineCoeffs LineDetector::line_fit(const Segment &segment, size_t it_start, size_t it_end) const noexcept {
    size_t len = it_end - it_start + 1;

    if (len < 2) {
        return LineCoeffs{0, 0, 0, DOUBLE_MAX};
    }

    cv::Mat design_mat = cv::Mat::zeros(3, 3, CV_64F);
    for (size_t it_curr = it_start; it_curr <= it_end; it_curr++) {
        const auto &x = segment[it_curr].x;
        const auto &y = segment[it_curr].y;

        design_mat.at<double>(0, 0) += x * x;
        design_mat.at<double>(0, 1) += x * y;
        design_mat.at<double>(0, 2) += x;
        design_mat.at<double>(1, 0) += x * y;
        design_mat.at<double>(1, 1) += y * y;
        design_mat.at<double>(1, 2) += y;
        design_mat.at<double>(2, 0) += x;
        design_mat.at<double>(2, 1) += y;
        design_mat.at<double>(2, 2) += 1;
    }

    cv::SVD svd(design_mat, cv::SVD::FULL_UV);
    cv::Mat solution = svd.vt.row(2);

    double A = solution.at<double>(0);
    double B = solution.at<double>(1);
    double C = solution.at<double>(2);

    double norm = std::sqrt(A * A + B * B);
    if (norm < EPSILON) {
        return LineCoeffs{0, 0, 0, DOUBLE_MAX};
    }

    A /= norm;
    B /= norm;
    C /= norm;

    double mean_error = 0.0;
    for (size_t it_curr = it_start; it_curr <= it_end; it_curr++) {
        double x = segment[it_curr].x;
        double y = segment[it_curr].y;
        mean_error += std::abs(A * x + B * y + C);
    }
    mean_error /= len;

    return {A, B, C, mean_error};
}


cv::Point2d
LineDetector::projection_point_to_line(const LineCoeffs &line_coeffs, const cv::Point2i &p, bool norm) const noexcept {
    const double &A = line_coeffs[0];
    const double &B = line_coeffs[1];
    const double &C = line_coeffs[2];

    if (norm) {
        return -(A * p.x + B * p.y + C) * cv::Point2d(A, B) + cv::Point2d(p.x, p.y);
    } else {
        double t = (A * p.x + B * p.y + C) / (A * A + B * B);
        return -t * cv::Point2d(A, B) + cv::Point2d(p.x, p.y);
    }
}


bool LineDetector::validate_line_on_segment(const cv::Mat &img, const Line &line, const Segment &segment) const {
    bool valid = false;

    if (line.get_len() >= 80) {
        valid = true;
    } else if (line.get_len() <= 25) {
        valid = validate_line_on_segment_in_rect(img, segment, line, true);
    } else {
        valid = validate_line_on_segment_in_rect(img, segment, line, false);
        if (valid == false) {
            valid = validate_line_on_segment_in_rect(img, segment, line, true);
        }
    }

    return valid;
}


bool LineDetector::validate_line_on_segment_in_rect(
    const cv::Mat &img, const Segment &segment, const Line &line, bool use_rect
) const {
    int width = img.cols;
    int height = img.rows;

    double line_angle;
    double A = line.get_A();
    double B = line.get_B();
    if (std::abs(B) > std::abs(A)) {
        double k = -A / B;
        line_angle = atan(k);
    } else {
        double k = -B / A;
        line_angle = atan(1.0 / k);
    }

    std::vector<cv::Point2i> rect_points;
    const auto &points = [&]() -> const auto & {
        if (use_rect) {
            rect_points = std::move(enumerate_rect_points(line.get_start(), line.get_end()));
            return rect_points;
        } else {
            return segment;
        }
    }();

    int count = 0;
    int aligned = 0;

    for (int idx = 0; idx < points.size(); idx++) {
        int row = points[idx].y;
        int col = points[idx].x;

        if (row <= 0 || row >= height - 1 || col <= 0 || col >= width - 1) {
            continue;
        }

        count++;

        int com1 =
            static_cast<int>(img.at<uchar>(row + 1, col + 1)) - static_cast<int>(img.at<uchar>(row - 1, col - 1));
        int com2 =
            static_cast<int>(img.at<uchar>(row - 1, col + 1)) - static_cast<int>(img.at<uchar>(row + 1, col - 1));

        int gx =
            com1 + com2 + static_cast<int>(img.at<uchar>(row, col + 1)) - static_cast<int>(img.at<uchar>(row, col - 1));
        int gy =
            com1 - com2 + static_cast<int>(img.at<uchar>(row + 1, col)) - static_cast<int>(img.at<uchar>(row - 1, col));

        double pixel_angle = nfa->atan2(static_cast<double>(gx), static_cast<double>(-gy));

        double diff = fabs(line_angle - pixel_angle);

        if (diff <= ANGLE_TOLERANCE || diff >= M_PI - ANGLE_TOLERANCE) {
            aligned++;
        }
    }

    return nfa->check_validation(count, aligned);
}


std::vector<cv::Point2i>
LineDetector::enumerate_rect_points(const cv::Point2d &start, const cv::Point2d &end, size_t width) const {
    std::vector<cv::Point2i> rect_points;
    cv::Point2d tmp_vertex[4];
    cv::Point2d vertex[4];

    cv::Point2d vec(end - start);
    vec /= cv::norm(vec);
    cv::Point2d normal(-vec.y, vec.x);

    double real_width = static_cast<double>(width) / 2.0;
    tmp_vertex[0] = start + normal * real_width;
    tmp_vertex[1] = end + normal * real_width;
    tmp_vertex[2] = end - normal * real_width;
    tmp_vertex[3] = start - normal * real_width;

    size_t offset = 3;
    if (start.x < end.x && start.y <= end.y) {
        offset = 0;
    } else if (start.x >= end.x && start.y < end.y) {
        offset = 1;
    } else if (start.x > end.x && start.y >= end.y) {
        offset = 2;
    }

    for (size_t idx = 0; idx < 4; idx++) {
        vertex[idx] = tmp_vertex[(offset + idx) % 4];
    }

    cv::Point2i curr_point(static_cast<int>(ceil(vertex[0].x) - 1), static_cast<int>(ceil(vertex[0].y)));
    double y_start = -DOUBLE_MAX, y_end = -DOUBLE_MAX;

    const double eps = 1e-2;
    while (true) {
        curr_point.y++;

        while (curr_point.y > y_end && curr_point.x <= vertex[2].x) {
            curr_point.x++;

            if (curr_point.x > vertex[2].x) {
                break;
            }


            if (static_cast<double>(curr_point.x) < vertex[3].x) {
                if (std::abs(vertex[0].x - vertex[3].x) <= eps) {
                    y_start = std::min(vertex[0].y, vertex[3].y);
                } else {
                    double t = (static_cast<double>(curr_point.x) - vertex[0].x) / (vertex[3].x - vertex[0].x);
                    y_start = vertex[0].y + t * (vertex[3].y - vertex[0].y);
                }
            } else {
                if (std::abs(vertex[3].x - vertex[2].x) <= eps) {
                    y_start = std::min(vertex[3].y, vertex[2].y);
                } else {
                    double t = (static_cast<double>(curr_point.x) - vertex[3].x) / (vertex[2].x - vertex[3].x);
                    y_start = vertex[3].y + t * (vertex[2].y - vertex[3].y);
                }
            }

            if (static_cast<double>(curr_point.x) < vertex[1].x) {
                if (std::abs(vertex[0].x - vertex[1].x) <= eps) {
                    y_end = std::max(vertex[0].y, vertex[1].y);
                } else {
                    double t = (static_cast<double>(curr_point.x) - vertex[0].x) / (vertex[1].x - vertex[0].x);
                    y_end = vertex[0].y + t * (vertex[1].y - vertex[0].y);
                }
            } else {
                if (std::abs(vertex[1].x - vertex[2].x) <= eps) {
                    y_end = std::max(vertex[1].y, vertex[2].y);
                } else {
                    double t = (static_cast<double>(curr_point.x) - vertex[1].x) / (vertex[2].x - vertex[1].x);
                    y_end = vertex[1].y + t * (vertex[2].y - vertex[1].y);
                }
            }

            curr_point.y = static_cast<int>(std::ceil(y_start));
        }

        if (curr_point.x > vertex[2].x) {
            break;
        }

        rect_points.emplace_back(curr_point.x, curr_point.y);
    }

    return rect_points;
}


}; // namespace ccl