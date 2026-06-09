#include "../../include/ccl/utility/SyntheticDataGenerator.hpp"
#include "../../include/ccl/utility/DetectionIO.hpp"
#include "../../include/ccl/utility/GridGenerator.hpp"
#include <algorithm>
#include <cmath>
#include <optional>
#include <random>
#include <string>


namespace ccl {


void SyntheticData::save_to_yaml(const std::string &filename, bool use_noise_point) const {
    std::vector<std::optional<cv::Point2d>> opt_pixels;
    if (!use_noise_point) {
        for (const auto &pt : ideal_points) {
            opt_pixels.push_back(pt);
        }
    } else {
        for (const auto &pt : noisy_points) {
            opt_pixels.push_back(pt);
        }
    }
    ccl::utils::save_detection_data_yaml(filename, opt_pixels, object_points, board_size, board_step, image_size);
}


cv::Mat SyntheticData::visualize(
    size_t marker_radius, size_t marker_thickness, MarkerType marker, cv::Scalar marker_color
) const {
    cv::Mat img(image_size, CV_8UC3, cv::Scalar(0, 0, 0));

    const cv::Scalar ideal_color = marker_color;
    const cv::Scalar noisy_color(0, 0, 255);
    const cv::Scalar grid_color(255, 0, 0);

    const int board_width = board_size.get_width();
    const int board_height = board_size.get_height();
    const int total = static_cast<int>(ideal_points.size());

    auto isValid = [](const cv::Point2d &p) {
        return p.x >= 0 && p.y >= 0;
    };

    auto drawMarkerFunc =
        [&](cv::Mat &img, const cv::Point &p, const cv::Scalar &color, int radius, int thickness, MarkerType type) {
            switch (type) {
            case MarkerType::CIRCLE:
                cv::circle(img, p, radius, color, thickness);
                break;
            case MarkerType::CROSS:
                cv::drawMarker(img, p, color, cv::MARKER_CROSS, radius * 2, thickness);
                break;
            case MarkerType::SQUARE:
                cv::drawMarker(img, p, color, cv::MARKER_SQUARE, radius * 2, thickness);
                break;
            case MarkerType::DIAMOND:
                cv::drawMarker(img, p, color, cv::MARKER_DIAMOND, radius * 2, thickness);
                break;
            }
        };

    for (int row = 0; row < board_height; ++row) {
        for (int col = 0; col < board_width - 1; ++col) {
            int idx1 = row * board_width + col;
            int idx2 = row * board_width + col + 1;
            if (idx1 < total && idx2 < total && isValid(ideal_points[idx1]) && isValid(ideal_points[idx2])) {
                cv::line(img, ideal_points[idx1], ideal_points[idx2], grid_color, 1);
            }
        }
    }

    for (int col = 0; col < board_width; ++col) {
        for (int row = 0; row < board_height - 1; ++row) {
            int idx1 = row * board_width + col;
            int idx2 = (row + 1) * board_width + col;
            if (idx1 < total && idx2 < total && isValid(ideal_points[idx1]) && isValid(ideal_points[idx2])) {
                cv::line(img, ideal_points[idx1], ideal_points[idx2], grid_color, 1);
            }
        }
    }

    for (const auto &pt : ideal_points) {
        if (pt.x < 0 || pt.y < 0) {
            continue;
        }
        cv::Point p(static_cast<int>(pt.x), static_cast<int>(pt.y));
        drawMarkerFunc(
            img, p, ideal_color, static_cast<int>(marker_radius), static_cast<int>(marker_thickness), marker
        );
    }

    for (const auto &pt : noisy_points) {
        if (pt.x < 0 || pt.y < 0) {
            continue;
        }
        cv::Point p(static_cast<int>(pt.x), static_cast<int>(pt.y));
        drawMarkerFunc(
            img, p, noisy_color, static_cast<int>(marker_radius), static_cast<int>(marker_thickness), marker
        );
    }

    cv::putText(
        img,
        "Frame " + std::to_string(frame_id),
        cv::Point(10, 30),
        cv::FONT_HERSHEY_SIMPLEX,
        1.0,
        cv::Scalar(255, 255, 255),
        2
    );

    return img;
}


bool SyntheticDataGenerator::Config::validate(std::string &error_msg) const {
    if (board_size.get_width() <= 0 || board_size.get_height() <= 0) {
        error_msg = "board_size must have positive dimensions";
        return false;
    }
    if (board_step <= 0.0) {
        error_msg = "square_size must be positive";
        return false;
    }
    if (image_size.width <= 0 || image_size.height <= 0) {
        error_msg = "image_size must be positive";
        return false;
    }
    if (noise_stddev < 0.0) {
        error_msg = "noise_stddev must be non-negative";
        return false;
    }
    if (outlier_ratio < 0.0 || outlier_ratio > 1.0) {
        error_msg = "outlier_ratio must be in [0,1]";
        return false;
    }
    if (outlier_frame_prob < 0.0 || outlier_frame_prob > 1.0) {
        error_msg = "outlier_frame_prob must be in [0,1]";
        return false;
    }
    if (outlier_min_amplitude <= 0.0 || outlier_max_amplitude <= outlier_min_amplitude) {
        error_msg = "outlier amplitude range invalid";
        return false;
    }
    if (num_views <= 0) {
        error_msg = "num_views must be positive";
        return false;
    }
    if (pose_range.angle_min >= pose_range.angle_max || pose_range.dist_min >= pose_range.dist_max ||
        pose_range.shift_x_min >= pose_range.shift_x_max || pose_range.shift_y_min >= pose_range.shift_y_max) {
        error_msg = "pose_range intervals must have min < max";
        return false;
    }
    return true;
}


SyntheticDataGenerator::SyntheticDataGenerator(std::unique_ptr<CameraModel> &&camera)
    : camera(std::move(camera)),
      rng(std::random_device{}()) {
    if (!this->camera) {
        throw SyntheticDataGeneratorException("Camera pointer is null");
    }
    std::string err;
    if (!config.validate(err)) {
        throw SyntheticDataGeneratorException("Config validation failed: " + err);
    }
    if (!this->camera->get_intrins().is_valid(config.image_size)) {
        throw SyntheticDataGeneratorException("Intrinsic params is not valid for the given image size");
    }
}


bool SyntheticDataGenerator::is_grid_convex(const std::vector<cv::Point2d> &points, int cols, int rows) const noexcept {
    if (points.size() != static_cast<size_t>(cols * rows)) {
        return false;
    }

    for (int r = 0; r < rows - 1; ++r) {
        for (int c = 0; c < cols - 1; ++c) {
            int tl = r * cols + c;
            int tr = r * cols + c + 1;
            int bl = (r + 1) * cols + c;
            int br = (r + 1) * cols + c + 1;

            auto cross = [](const cv::Point2d &a, const cv::Point2d &b, const cv::Point2d &c) {
                return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
            };

            double s1 = cross(points[tl], points[tr], points[bl]);
            double s2 = cross(points[tl], points[tr], points[br]);
            double s3 = cross(points[bl], points[br], points[tl]);
            double s4 = cross(points[bl], points[br], points[tr]);

            if (!((s1 * s2 > 0) && (s3 * s4 > 0))) {
                return false;
            }
        }
    }
    return true;
}


bool SyntheticDataGenerator::set_new_pose(
    const ExtrinsicParams &pose, const std::vector<cv::Point3d> &board_pts
) const {
    if (!camera->set_pose(pose)) {
        return false;
    }
    auto proj = camera->project(board_pts);

    for (const auto &p : proj) {
        if (p.x < 0 || p.x >= config.image_size.width || p.y < 0 || p.y >= config.image_size.height) {
            return false;
        }
    }
    return is_grid_convex(
        proj, static_cast<int>(config.board_size.get_width()), static_cast<int>(config.board_size.get_height())
    );
}


ExtrinsicParams SyntheticDataGenerator::generate_random_pose() const {
    const auto &range = config.pose_range;
    std::uniform_real_distribution<> angle_x(range.angle_min, range.angle_max);
    std::uniform_real_distribution<> angle_y(range.angle_min, range.angle_max);
    std::uniform_real_distribution<> angle_z(range.angle_min, range.angle_max);
    std::uniform_real_distribution<> shift_x(range.shift_x_min, range.shift_x_max);
    std::uniform_real_distribution<> shift_y(range.shift_y_min, range.shift_y_max);
    std::uniform_real_distribution<> dist_z(range.dist_min, range.dist_max);

    cv::Vec3d rvec(angle_x(rng), angle_y(rng), angle_z(rng));
    cv::Vec3d tvec(shift_x(rng), shift_y(rng), dist_z(rng));
    return ExtrinsicParams(rvec, tvec);
}


cv::Point2d SyntheticDataGenerator::add_noise(const cv::Point2d &pt) const {
    std::normal_distribution<> noise(0.0, config.noise_stddev);
    const int max_attempts = 100;
    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        double nx = pt.x + noise(rng);
        double ny = pt.y + noise(rng);
        if (nx >= 0 && nx < config.image_size.width && ny >= 0 && ny < config.image_size.height) {
            return {nx, ny};
        }
    }
    double nx = std::clamp(pt.x + noise(rng), 0.0, static_cast<double>(config.image_size.width - 1));
    double ny = std::clamp(pt.y + noise(rng), 0.0, static_cast<double>(config.image_size.height - 1));
    return {nx, ny};
}


void SyntheticDataGenerator::add_noise_and_outliers(
    const std::vector<cv::Point2d> &ideal, std::vector<cv::Point2d> &noisy, bool has_outliers
) const {
    noisy.resize(ideal.size());
    std::uniform_real_distribution<> outlier_prob(0.0, 1.0);
    std::uniform_real_distribution<> outlier_angle(0.0, 2 * M_PI);
    std::uniform_real_distribution<> outlier_mag(config.outlier_min_amplitude, config.outlier_max_amplitude);

    for (size_t i = 0; i < ideal.size(); ++i) {
        cv::Point2d p = add_noise(ideal[i]);

        if (has_outliers && outlier_prob(rng) < config.outlier_ratio) {
            double angle = outlier_angle(rng);
            double mag = outlier_mag(rng);
            p.x += mag * std::cos(angle);
            p.y += mag * std::sin(angle);
            p.x = std::max(0.0, std::min(static_cast<double>(config.image_size.width - 1), p.x));
            p.y = std::max(0.0, std::min(static_cast<double>(config.image_size.height - 1), p.y));
        }
        noisy[i] = p;
    }
}


std::vector<SyntheticData> SyntheticDataGenerator::generate() {
    std::vector<SyntheticData> frames;
    frames.reserve(config.num_views);

    auto obj_pts = utils::generate_grid(config.board_size, config.board_step, config.board_origin);
    std::uniform_real_distribution<> frame_outlier_prob(0.0, 1.0);

    for (int i = 0; i < config.num_views; ++i) {
        ExtrinsicParams pose;
        bool valid = false;
        const int max_attempts = 500;
        for (int attempt = 0; attempt < max_attempts; ++attempt) {
            pose = generate_random_pose();
            if (set_new_pose(pose, obj_pts)) {
                valid = true;
                break;
            }
        }
        if (!valid) {
            throw SyntheticDataGeneratorException(
                "Failed to generate a valid camera pose after " + std::to_string(max_attempts) + " attempts."
            );
        }

        auto ideal = camera->project(obj_pts);

        bool has_outliers = (frame_outlier_prob(rng) < config.outlier_frame_prob);
        std::vector<cv::Point2d> noisy;
        add_noise_and_outliers(ideal, noisy, has_outliers);

        frames.push_back(
            {i,
             config.image_size,
             config.board_size,
             config.board_step,
             obj_pts,
             std::move(ideal),
             std::move(noisy),
             pose}
        );
    }

    return frames;
}


} // namespace ccl
