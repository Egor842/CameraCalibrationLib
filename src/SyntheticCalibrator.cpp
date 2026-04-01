#include "../include/SyntheticCalibrator.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <random>
#include <thread>


namespace ccl {


bool GenerationParams::validate(std::string &error_msg) const {
    if (!camera_model) {
        error_msg = "camera_model must be set";
        return false;
    }
    if (board_params.board_size.get_width() <= 0 || board_params.board_size.get_height() <= 0) {
        error_msg = "board_size must have positive dimensions";
        return false;
    }
    if (board_params.square_size <= 0) {
        error_msg = "square_size must be positive";
        return false;
    }
    if (image_size.width <= 120 || image_size.height <= 60) {
        error_msg = "image_size must be greater then (120, 60)";
        return false;
    }
    if (noise_sigma < 0) {
        error_msg = "noise_sigma must be non-negative";
        return false;
    }
    if (outlier_ratio < 0 || outlier_ratio > 1) {
        error_msg = "outlier_ratio must be in [0,1]";
        return false;
    }
    if (outlier_frame_prob < 0 || outlier_frame_prob > 1) {
        error_msg = "outlier_frame_prob must be in [0,1]";
        return false;
    }
    if (outlier_min_sigma <= 0 || outlier_max_sigma <= outlier_min_sigma) {
        error_msg = "outlier_min_sigma must be positive and less than outlier_max_sigma";
        return false;
    }
    if (num_frames <= 0) {
        error_msg = "num_frames must be positive";
        return false;
    }
    if (pose_range.angle_min >= pose_range.angle_max) {
        error_msg = "pose_range.angle_min must be less than angle_max";
        return false;
    }
    if (pose_range.dist_min >= pose_range.dist_max) {
        error_msg = "pose_range.dist_min must be less than dist_max";
        return false;
    }
    if (pose_range.shift_x_min >= pose_range.shift_x_max || pose_range.shift_y_min >= pose_range.shift_y_max) {
        error_msg = "pose_range shift intervals must have min < max";
        return false;
    }
    return true;
}


std::vector<cv::Point3d> SyntheticCalibrator::generate_board_points() const {
    std::vector<cv::Point3d> points;
    const auto &board = params_.board_params;
    for (int i = 0; i < board.board_size.get_height(); ++i) {
        for (int j = 0; j < board.board_size.get_width(); ++j) {
            double x = board.origin.x + j * board.square_size;
            double y = board.origin.y + i * board.square_size;
            points.emplace_back(x, y, board.origin.z);
        }
    }
    return points;
}


bool SyntheticCalibrator::is_grid_convex(
    const std::vector<cv::Point2d> &points, size_t cols, size_t rows
) const noexcept {
    for (int row = 0; row < rows - 1; ++row) {
        for (int col = 0; col < cols - 1; ++col) {
            int tl = row * cols + col;
            int tr = row * cols + col + 1;
            int bl = (row + 1) * cols + col;
            int br = (row + 1) * cols + col + 1;

            auto cross = [](const cv::Point2d &a, const cv::Point2d &b, const cv::Point2d &c) {
                return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
            };

            double s1 = cross(points[tl], points[tr], points[bl]);
            double s2 = cross(points[tl], points[tr], points[br]);
            double s3 = cross(points[bl], points[br], points[tl]);
            double s4 = cross(points[bl], points[br], points[tr]);

            if ((s1 * s2 > 0) && (s3 * s4 > 0)) {
                continue;
            }
            return false;
        }
    }
    return true;
}


bool SyntheticCalibrator::generate_valid_pose(
    cv::Mat &rvec, cv::Mat &tvec, const std::vector<cv::Point3d> &board_points
) const {
    std::random_device rd;
    std::mt19937 gen(rd());
    const auto &range = params_.pose_range;
    std::uniform_real_distribution<> angle_x(range.angle_min, range.angle_max);
    std::uniform_real_distribution<> angle_y(range.angle_min, range.angle_max);
    std::uniform_real_distribution<> angle_z(range.angle_min, range.angle_max);
    std::uniform_real_distribution<> shift_x(range.shift_x_min, range.shift_x_max);
    std::uniform_real_distribution<> shift_y(range.shift_y_min, range.shift_y_max);
    std::uniform_real_distribution<> dist_z(range.dist_min, range.dist_max);

    const int max_attempts = 500;
    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        rvec = (cv::Mat_<double>(3, 1) << angle_x(gen), angle_y(gen), angle_z(gen));
        tvec = (cv::Mat_<double>(3, 1) << shift_x(gen), shift_y(gen), dist_z(gen));

        params_.camera_model->set_external(ExtrinsicParams(
            cv::Vec3d(rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2)),
            cv::Vec3d(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2))
        ));

        auto proj_points = params_.camera_model->project_points(board_points);
        bool all_inside = true;
        for (const auto &p : proj_points) {
            if (p.x < 0 || p.x >= params_.image_size.width || p.y < 0 || p.y >= params_.image_size.height) {
                all_inside = false;
                break;
            }
        }
        if (all_inside &&
            is_grid_convex(
                proj_points, params_.board_params.board_size.get_width(), params_.board_params.board_size.get_height()
            )) {
            return true;
        }
    }
    std::cout << "GG" << std::endl;
    return false;
}


SyntheticCalibrator::SyntheticCalibrator(const GenerationParams &params)
    : params_(params),
      stop_visualization_(false),
      visualization_ready_(false) {
    std::string err;
    if (!params_.validate(err)) {
        throw SyntheticCalibratorException("Validation failed: " + err);
    }
}


void SyntheticCalibrator::generate_data() {
    const int nframes = params_.num_frames;
    const auto board_points = generate_board_points();

    object_points_.assign(nframes, board_points);
    ideal_points_.resize(nframes);
    noisy_points_.resize(nframes);
    rvecs_true_.resize(nframes);
    tvecs_true_.resize(nframes);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> noise(0.0, params_.noise_sigma);
    std::uniform_real_distribution<> outlier_prob(0.0, 1.0);
    std::uniform_real_distribution<> outlier_mag(params_.outlier_min_sigma, params_.outlier_max_sigma);
    std::uniform_real_distribution<> outlier_angle(0.0, 2 * M_PI);

    for (int i = 0; i < nframes; ++i) {
        cv::Mat rvec, tvec;
        if (!generate_valid_pose(rvec, tvec, board_points)) {
            throw SyntheticCalibratorException("Calibration failed: can not generate valid camera poses");
        }
        rvecs_true_[i] = rvec.clone();
        tvecs_true_[i] = tvec.clone();

        params_.camera_model->set_external(ExtrinsicParams(
            cv::Vec3d(rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2)),
            cv::Vec3d(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2))
        ));

        auto proj_points = params_.camera_model->project_points(board_points);
        ideal_points_[i].resize(proj_points.size());
        for (size_t j = 0; j < proj_points.size(); ++j) {
            ideal_points_[i][j] = proj_points[j];
        }

        bool has_outliers = (outlier_prob(gen) < params_.outlier_frame_prob);
        int total_points = ideal_points_[i].size();
        noisy_points_[i].resize(total_points);

        for (int j = 0; j < total_points; ++j) {
            cv::Point2d p_ideal = ideal_points_[i][j];
            cv::Point2d p_noisy = p_ideal + cv::Point2d(noise(gen), noise(gen));

            if (has_outliers && (outlier_prob(gen) < params_.outlier_ratio)) {
                double magnitude = outlier_mag(gen) * params_.noise_sigma;
                double angle = outlier_angle(gen);
                p_noisy += cv::Point2d(magnitude * std::cos(angle), magnitude * std::sin(angle));
            }

            p_noisy.x = std::max(0.0, std::min(static_cast<double>(params_.image_size.width - 1), p_noisy.x));
            p_noisy.y = std::max(0.0, std::min(static_cast<double>(params_.image_size.height - 1), p_noisy.y));
            noisy_points_[i][j] = p_noisy;
        }
    }
}


std::vector<std::vector<cv::Point3d>> SyntheticCalibrator::get_object_points_3d() const {
    return object_points_;
}


std::vector<std::vector<cv::Point2d>> SyntheticCalibrator::get_image_points_2d() const {
    return noisy_points_;
}


void SyntheticCalibrator::visualization_thread() {
    if (!params_.visualize) {
        return;
    }

    cv::namedWindow("Synthetic Data", cv::WINDOW_NORMAL);
    cv::resizeWindow("Synthetic Data", params_.image_size.width, params_.image_size.height);

    int board_width = params_.board_params.board_size.get_width();
    int board_height = params_.board_params.board_size.get_height();

    while (!stop_visualization_) {
        std::unique_lock<std::mutex> lock(vis_mutex_);
        vis_cv_.wait(lock, [this] {
            return visualization_ready_ || stop_visualization_;
        });
        if (stop_visualization_) {
            break;
        }

        for (size_t frame_idx = 0; frame_idx < noisy_points_.size(); ++frame_idx) {
            cv::Mat img = cv::Mat::zeros(params_.image_size, CV_8UC3);

            const auto &ideal = ideal_points_[frame_idx];
            const auto &noisy = noisy_points_[frame_idx];

            for (const auto &pt : ideal) {
                cv::circle(img, pt, 2, cv::Scalar(0, 255, 0), -1);
            }
            for (const auto &pt : noisy) {
                cv::circle(img, pt, 2, cv::Scalar(0, 0, 255), -1);
            }

            for (int row = 0; row < board_height; ++row) {
                for (int col = 0; col < board_width - 1; ++col) {
                    int idx1 = row * board_width + col;
                    int idx2 = row * board_width + col + 1;
                    if (idx1 < ideal.size() && idx2 < ideal.size()) {
                        cv::line(img, ideal[idx1], ideal[idx2], cv::Scalar(255, 0, 0), 1);
                    }
                }
            }
            for (int col = 0; col < board_width; ++col) {
                for (int row = 0; row < board_height - 1; ++row) {
                    int idx1 = row * board_width + col;
                    int idx2 = (row + 1) * board_width + col;
                    if (idx1 < ideal.size() && idx2 < ideal.size()) {
                        cv::line(img, ideal[idx1], ideal[idx2], cv::Scalar(255, 0, 0), 1);
                    }
                }
            }

            cv::putText(
                img,
                "Frame " + std::to_string(frame_idx),
                cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX,
                1.0,
                cv::Scalar(255, 255, 255),
                2
            );
            cv::imshow("Synthetic Data", img);
            cv::waitKey(0);
        }
        visualization_ready_ = false;
    }
    cv::destroyWindow("Synthetic Data");
}


void SyntheticCalibrator::save_to_files() const {
    if (!params_.save_to_files) {
        return;
    }

    std::string ideal_path = params_.output_dir + "/ideal_points.txt";
    std::string noisy_path = params_.output_dir + "/noisy_points.txt";

    std::ofstream f_ideal(ideal_path), f_noisy(noisy_path);
    if (!f_ideal.is_open() || !f_noisy.is_open()) {
        std::cerr << "Failed to open output files in " << params_.output_dir << std::endl;
        return;
    }

    for (size_t i = 0; i < ideal_points_.size(); ++i) {
        f_ideal << "Frame " << i << "\n";
        f_noisy << "Frame " << i << "\n";
        for (size_t j = 0; j < ideal_points_[i].size(); ++j) {
            f_ideal << ideal_points_[i][j].x << " " << ideal_points_[i][j].y << "\n";
            f_noisy << noisy_points_[i][j].x << " " << noisy_points_[i][j].y << "\n";
        }
    }
}


void SynthticCalibrationResult::compute_errors(
    const IntrinsicParams &true_intrinsic,
    const std::vector<double> &true_dist,
    const std::vector<ExtrinsicParams> &true_extrinsics
) {
    fx_error = intrinsic.fx_error(true_intrinsic);
    fy_error = intrinsic.fy_error(true_intrinsic);
    cx_error = intrinsic.cx_error(true_intrinsic);
    cy_error = intrinsic.cy_error(true_intrinsic);
    skew_error = intrinsic.skew_error(true_intrinsic);

    dist_coeffs_errors.clear();
    auto dist_coeffs_estimated = distortion->get_coefficients();
    size_t n = std::min(dist_coeffs_estimated.size(), true_dist.size());
    dist_coeffs_errors.resize(n);
    for (size_t i = 0; i < n; ++i) {
        dist_coeffs_errors[i] = std::abs(dist_coeffs_estimated[i] - true_dist[i]);
    }

    rotation_errors.clear();
    translation_errors.clear();
    size_t n_views = std::min(extrinsics_vec.size(), true_extrinsics.size());
    rotation_errors.reserve(n_views);
    translation_errors.reserve(n_views);

    for (size_t i = 0; i < n_views; ++i) {
        cv::Vec3d rvec = extrinsics_vec[i].get_rvec();
        cv::Vec3d rvec_true = true_extrinsics[i].get_tvec();
        cv::Vec3d rvec_error = rvec_true - rvec;
        rotation_errors[n_views][0] = std::fabs(rvec_error[0]);
        rotation_errors[n_views][1] = std::fabs(rvec_error[1]);
        rotation_errors[n_views][2] = std::fabs(rvec_error[2]);

        cv::Vec3d tvec = extrinsics_vec[i].get_tvec();
        cv::Vec3d tvec_true = true_extrinsics[i].get_tvec();
        cv::Vec3d tvec_error = tvec_true - tvec;
        translation_errors[n_views][0] = std::fabs(tvec_error[0]);
        translation_errors[n_views][1] = std::fabs(tvec_error[1]);
        translation_errors[n_views][2] = std::fabs(tvec_error[2]);
    }
}


void SynthticCalibrationResult::assign_base(CalibrationResult &&base) {
    this->intrinsic = std::move(base.intrinsic);
    this->extrinsics_vec = std::move(base.extrinsics_vec);
    this->distortion = std::move(base.distortion);
    this->successfully = base.successfully;
    this->rmse = base.rmse;
    this->time_seconds = base.time_seconds;
    this->per_view_errors = std::move(base.per_view_errors);
}


SynthticCalibrationResult
SyntheticCalibrator::run(const std::unique_ptr<Calibrator> &calibrator, bool generate_new_data) {
    if (generate_new_data || ideal_points_.empty()) {
        generate_data();
    }

    if (params_.visualize) {
        stop_visualization_ = false;
        visualization_ready_ = true;
        vis_thread_ = std::thread(&SyntheticCalibrator::visualization_thread, this);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    SynthticCalibrationResult result;
    auto raw_result = calibrator->calibrate(get_object_points_3d(), get_image_points_2d());
    if (raw_result->successfully) {
        result.assign_base(std::move(*raw_result));
    } else {
        throw SyntheticCalibratorException("Calibration failed");
    }

    std::vector<ExtrinsicParams> true_extrinsics;
    true_extrinsics.reserve(rvecs_true_.size());
    for (size_t i = 0; i < rvecs_true_.size(); ++i) {
        true_extrinsics.emplace_back(
            cv::Vec3d(rvecs_true_[i].at<double>(0), rvecs_true_[i].at<double>(1), rvecs_true_[i].at<double>(2)),
            cv::Vec3d(tvecs_true_[i].at<double>(0), tvecs_true_[i].at<double>(1), tvecs_true_[i].at<double>(2))
        );
    }

    result.compute_errors(
        params_.camera_model->get_intrisic(),
        params_.camera_model->get_distortion()->get_coefficients(),
        true_extrinsics
    );

    save_to_files();

    if (params_.visualize) {
        stop_visualization_ = true;
        vis_cv_.notify_all();
        if (vis_thread_.joinable()) {
            vis_thread_.join();
        }
    }

    return result;
}


} // namespace ccl
