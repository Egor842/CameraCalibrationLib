#include "../include/BrownConradyCalibrator.hpp"
#include <ceres/ceres.h>
#include <ceres/loss_function.h>
#include <ceres/rotation.h>
#include <chrono>


namespace ccl {


namespace {


struct ReprojectionErrorCeres {
    ReprojectionErrorCeres(
        double observed_x, double observed_y, double point_3d_x, double point_3d_y, double point_3d_z
    )
        : observed_x_(observed_x),
          observed_y_(observed_y),
          point_3d_x_(point_3d_x),
          point_3d_y_(point_3d_y),
          point_3d_z_(point_3d_z) {}

    template <typename T>
    bool operator()(
        const T *const intrinsics, // [fx, fy, cx, cy, k1, k2, p1, p2, k3]
        const T *const rvec,       // rvec (3)
        const T *const tvec,       // tvec (3)
        T *residuals               // [res_x, res_y]
    ) const {
        const T &fx = intrinsics[0];
        const T &fy = intrinsics[1];
        const T &cx = intrinsics[2];
        const T &cy = intrinsics[3];
        const T &k1 = intrinsics[4];
        const T &k2 = intrinsics[5];
        const T &p1 = intrinsics[6];
        const T &p2 = intrinsics[7];
        const T &k3 = intrinsics[8];

        T R[9];
        ceres::AngleAxisToRotationMatrix(rvec, R);
        T Rt[9];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                Rt[i * 3 + j] = R[j * 3 + i];
            }
        }

        T X = Rt[0] * point_3d_x_ + Rt[1] * point_3d_y_ + Rt[2] * point_3d_z_ + tvec[0];
        T Y = Rt[3] * point_3d_x_ + Rt[4] * point_3d_y_ + Rt[5] * point_3d_z_ + tvec[1];
        T Z = Rt[6] * point_3d_x_ + Rt[7] * point_3d_y_ + Rt[8] * point_3d_z_ + tvec[2];

        if (Z <= T(0)) {
            residuals[0] = T(1e6);
            residuals[1] = T(1e6);
            return true;
        }

        T xp = X / Z;
        T yp = Y / Z;

        T r2 = xp * xp + yp * yp;
        T r4 = r2 * r2;
        T r6 = r4 * r2;
        T radial = T(1) + k1 * r2 + k2 * r4 + k3 * r6;

        T xpp = xp * radial + T(2) * p1 * xp * yp + p2 * (r2 + T(2) * xp * xp);
        T ypp = yp * radial + p1 * (r2 + T(2) * yp * yp) + T(2) * p2 * xp * yp;

        T predicted_x = fx * xpp + cx;
        T predicted_y = fy * ypp + cy;

        residuals[0] = predicted_x - T(observed_x_);
        residuals[1] = predicted_y - T(observed_y_);
        return true;
    }

private:
    const double observed_x_, observed_y_;
    const double point_3d_x_, point_3d_y_, point_3d_z_;
};


std::vector<double> mat_to_vector(const cv::Mat &mat) {
    std::vector<double> vec(mat.total());
    if (mat.isContinuous()) {
        std::memcpy(vec.data(), mat.data, mat.total() * sizeof(double));
    } else {
        for (int i = 0; i < mat.rows; ++i) {
            std::memcpy(vec.data() + i * mat.cols, mat.ptr<double>(i), mat.cols * sizeof(double));
        }
    }
    return vec;
}


cv::Matx33d
compute_homography_for_view(const cv::Mat &object_points, const cv::Mat &image_points, double cx, double cy) {
    cv::Mat homography_raw = cv::findHomography(object_points, image_points);
    cv::Matx33d homography;
    homography_raw.convertTo(homography, CV_64F);

    homography(0, 0) -= homography(2, 0) * cx;
    homography(0, 1) -= homography(2, 1) * cx;
    homography(0, 2) -= homography(2, 2) * cx;
    homography(1, 0) -= homography(2, 0) * cy;
    homography(1, 1) -= homography(2, 1) * cy;
    homography(1, 2) -= homography(2, 2) * cy;

    return homography;
}


void build_focal_length_equations(
    const cv::Matx33d &homography, cv::Mat_<double> &linear_system, cv::Mat_<double> &linear_rhs, int view_index
) {
    cv::Vec3d h, v, d1, d2;
    cv::Vec4d normalization(0.0, 0.0, 0.0, 0.0);

    for (int j = 0; j < 3; ++j) {
        const double t0 = homography(j, 0);
        const double t1 = homography(j, 1);

        h[j] = t0;
        v[j] = t1;
        d1[j] = (t0 + t1) * 0.5;
        d2[j] = (t0 - t1) * 0.5;

        normalization[0] += t0 * t0;
        normalization[1] += t1 * t1;
        normalization[2] += d1[j] * d1[j];
        normalization[3] += d2[j] * d2[j];
    }

    for (int j = 0; j < 4; ++j) {
        normalization[j] = 1.0 / std::sqrt(normalization[j]);
    }

    for (int j = 0; j < 3; ++j) {
        h[j] *= normalization[0];
        v[j] *= normalization[1];
        d1[j] *= normalization[2];
        d2[j] *= normalization[3];
    }

    linear_system(view_index * 2 + 0, 0) = h[0] * v[0];
    linear_system(view_index * 2 + 0, 1) = h[1] * v[1];
    linear_system(view_index * 2 + 1, 0) = d1[0] * d2[0];
    linear_system(view_index * 2 + 1, 1) = d1[1] * d2[1];
    linear_rhs(view_index * 2 + 0) = -h[2] * v[2];
    linear_rhs(view_index * 2 + 1) = -d1[2] * d2[2];
}


std::pair<double, double>
solve_focal_lengths(const cv::Mat_<double> &linear_system, const cv::Mat_<double> &linear_rhs) {
    cv::Vec2d focal_lengths;
    cv::solve(linear_system, linear_rhs, focal_lengths, cv::DECOMP_NORMAL + cv::DECOMP_SVD);

    const double fx = std::sqrt(std::abs(1.0 / focal_lengths[0]));
    const double fy = std::sqrt(std::abs(1.0 / focal_lengths[1]));

    return {fx, fy};
}


}; // namespace


BrownConradyCalibrator::PreparedData BrownConradyCalibrator::prepare_data(
    const std::vector<std::vector<cv::Point3d>> &object_points,
    const std::vector<std::vector<cv::Point2d>> &image_points
) const {
    PreparedData data;
    data.num_views = static_cast<int>(object_points.size());

    data.total_points = 0;
    for (const auto &view : object_points) {
        data.total_points += static_cast<int>(view.size());
    }

    data.all_object_points.create(1, data.total_points, CV_64FC3);
    data.all_image_points.create(1, data.total_points, CV_64FC2);
    data.points_per_view.create(1, data.num_views, CV_32SC1);

    int current_pos = 0;
    for (int i = 0; i < data.num_views; ++i) {
        const int num_points = static_cast<int>(object_points[i].size());
        data.points_per_view.at<int>(i) = num_points;

        for (int j = 0; j < num_points; ++j) {
            data.all_object_points.at<cv::Point3d>(current_pos + j) = object_points[i][j];
            data.all_image_points.at<cv::Point2d>(current_pos + j) = image_points[i][j];
        }
        current_pos += num_points;
    }

    return data;
}


BrownConradyCalibrator::IntrinsicInitialGuess
BrownConradyCalibrator::estimate_intrinsic_matrix(const PreparedData &data) const {
    const double cx = (image_size.width - 1) * 0.5;
    const double cy = (image_size.height - 1) * 0.5;

    cv::Mat_<double> linear_system(2 * data.num_views, 2);
    cv::Mat_<double> linear_rhs(2 * data.num_views, 1);

    int current_pos = 0;
    for (int i = 0; i < data.num_views; ++i) {
        const int num_points = data.points_per_view.at<int>(i);

        cv::Mat view_object_points = data.all_object_points.colRange(current_pos, current_pos + num_points);
        cv::Mat view_image_points = data.all_image_points.colRange(current_pos, current_pos + num_points);
        current_pos += num_points;

        auto homography = compute_homography_for_view(view_object_points, view_image_points, cx, cy);

        build_focal_length_equations(homography, linear_system, linear_rhs, i);
    }

    auto [fx, fy] = solve_focal_lengths(linear_system, linear_rhs);

    cv::Mat camera_matrix = cv::Mat::eye(3, 3, CV_64F);
    camera_matrix.at<double>(0, 0) = fx;
    camera_matrix.at<double>(1, 1) = fy;
    camera_matrix.at<double>(0, 2) = cx;
    camera_matrix.at<double>(1, 2) = cy;

    return IntrinsicInitialGuess{std::move(camera_matrix), fx, fy, cx, cy};
}


std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>> BrownConradyCalibrator::estimate_rotation_and_translation(
    const PreparedData &data, const cv::Mat &camera_matrix, const cv::Mat &distortion_coeffs
) const {
    const int num_views = data.num_views;
    std::vector<cv::Mat> rotation_vectors(num_views);
    std::vector<cv::Mat> translation_vectors(num_views);

    int current_pos = 0;
    for (int i = 0; i < num_views; ++i) {
        const int num_points = data.points_per_view.at<int>(i);

        cv::Mat view_object_points = data.all_object_points.colRange(current_pos, current_pos + num_points);
        cv::Mat view_image_points = data.all_image_points.colRange(current_pos, current_pos + num_points);

        rotation_vectors[i].create(3, 1, CV_64F);
        translation_vectors[i].create(3, 1, CV_64F);

        cv::solvePnP(
            view_object_points,
            view_image_points,
            camera_matrix,
            distortion_coeffs,
            rotation_vectors[i],
            translation_vectors[i],
            false, // useExtrinsicGuess = false
            cv::SOLVEPNP_ITERATIVE
        );

        current_pos += num_points;
    }

    return {rotation_vectors, translation_vectors};
}


BrownConradyCalibrator::InitialGuess BrownConradyCalibrator::compute_initial_guess(
    const std::vector<std::vector<cv::Point3d>> &object_points,
    const std::vector<std::vector<cv::Point2d>> &image_points
) const {
    auto prepared_data = prepare_data(object_points, image_points);

    auto intrinsic_guess = estimate_intrinsic_matrix(prepared_data);

    cv::Mat distortion_coeffs = cv::Mat::zeros(1, 5, CV_64F);

    auto [rotation_vectors, translation_vectors] =
        estimate_rotation_and_translation(prepared_data, intrinsic_guess.camera_matrix, distortion_coeffs);

    return InitialGuess{
        std::move(intrinsic_guess.camera_matrix),
        std::move(distortion_coeffs),
        std::move(rotation_vectors),
        std::move(translation_vectors)
    };
}


BrownConradyCalibrationResult BrownConradyCalibrator::calibrate_brown_conrady(
    const std::vector<std::vector<cv::Point3d>> &object_points,
    const std::vector<std::vector<cv::Point2d>> &image_points
) const {
    auto start_time = std::chrono::high_resolution_clock::now();

    auto initial_guess = compute_initial_guess(object_points, image_points);

    double intrinsics[9];
    intrinsics[0] = initial_guess.camera_matrix.at<double>(0, 0);  // fx
    intrinsics[1] = initial_guess.camera_matrix.at<double>(1, 1);  // fy
    intrinsics[2] = initial_guess.camera_matrix.at<double>(0, 2);  // cx
    intrinsics[3] = initial_guess.camera_matrix.at<double>(1, 2);  // cy
    intrinsics[4] = initial_guess.distortion_coeffs.at<double>(0); // k1
    intrinsics[5] = initial_guess.distortion_coeffs.at<double>(1); // k2
    intrinsics[6] = initial_guess.distortion_coeffs.at<double>(2); // p1
    intrinsics[7] = initial_guess.distortion_coeffs.at<double>(3); // p2
    intrinsics[8] = initial_guess.distortion_coeffs.at<double>(4); // k3

    const int num_views = static_cast<int>(initial_guess.rotation_vectors.size());

    std::vector<std::vector<double>> rvecs_data(num_views, std::vector<double>(3));
    std::vector<std::vector<double>> tvecs_data(num_views, std::vector<double>(3));

    for (int i = 0; i < num_views; ++i) {
        for (int j = 0; j < 3; ++j) {
            rvecs_data[i][j] = initial_guess.rotation_vectors[i].at<double>(j);
            tvecs_data[i][j] = initial_guess.translation_vectors[i].at<double>(j);
        }
    }

    ceres::Problem problem;

    for (int view = 0; view < num_views; ++view) {
        const int num_points = static_cast<int>(object_points[view].size());
        for (int point = 0; point < num_points; ++point) {
            auto *cost = new ceres::AutoDiffCostFunction<ReprojectionErrorCeres, 2, 9, 3, 3>(new ReprojectionErrorCeres(
                image_points[view][point].x,
                image_points[view][point].y,
                object_points[view][point].x,
                object_points[view][point].y,
                object_points[view][point].z
            ));

            auto *loss = new ceres::CauchyLoss(1.0); // δ = 1 пиксель

            problem.AddResidualBlock(cost, loss, intrinsics, rvecs_data[view].data(), tvecs_data[view].data());
        }
    }

    // problem.SetParameterLowerBound(intrinsics, 0, 100.0);
    // problem.SetParameterUpperBound(intrinsics, 0, 5000.0);
    // problem.SetParameterLowerBound(intrinsics, 1, 100.0);
    // problem.SetParameterUpperBound(intrinsics, 1, 5000.0);

    problem.SetParameterLowerBound(intrinsics, 2, 0.0);
    problem.SetParameterUpperBound(intrinsics, 2, image_size.width);
    problem.SetParameterLowerBound(intrinsics, 3, 0.0);
    problem.SetParameterUpperBound(intrinsics, 3, image_size.height);

    // problem.SetParameterLowerBound(intrinsics, 4, -1.0); // k1
    // problem.SetParameterUpperBound(intrinsics, 4, 1.0);
    // problem.SetParameterLowerBound(intrinsics, 5, -1.0); // k2
    // problem.SetParameterUpperBound(intrinsics, 5, 1.0);
    // problem.SetParameterLowerBound(intrinsics, 6, -0.1); // p1
    // problem.SetParameterUpperBound(intrinsics, 6, 0.1);
    // problem.SetParameterLowerBound(intrinsics, 7, -0.1); // p2
    // problem.SetParameterUpperBound(intrinsics, 7, 0.1);
    // problem.SetParameterLowerBound(intrinsics, 8, -1.0); // k3
    // problem.SetParameterUpperBound(intrinsics, 8, 1.0);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.use_nonmonotonic_steps = true;
    options.max_consecutive_nonmonotonic_steps = 5;
    options.max_num_iterations = 500;
    options.num_threads = 4;
    options.function_tolerance = 1e-8;
    options.gradient_tolerance = 1e-12;
    options.parameter_tolerance = 1e-10;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    cv::Mat optimized_camera_matrix = cv::Mat::eye(3, 3, CV_64F);
    optimized_camera_matrix.at<double>(0, 0) = intrinsics[0];
    optimized_camera_matrix.at<double>(1, 1) = intrinsics[1];
    optimized_camera_matrix.at<double>(0, 2) = intrinsics[2];
    optimized_camera_matrix.at<double>(1, 2) = intrinsics[3];

    cv::Mat optimized_distortion = cv::Mat::zeros(1, 5, CV_64F);
    optimized_distortion.at<double>(0) = intrinsics[4];
    optimized_distortion.at<double>(1) = intrinsics[5];
    optimized_distortion.at<double>(2) = intrinsics[6];
    optimized_distortion.at<double>(3) = intrinsics[7];
    optimized_distortion.at<double>(4) = intrinsics[8];

    std::vector<cv::Mat> optimized_rvecs(num_views);
    std::vector<cv::Mat> optimized_tvecs(num_views);

    for (int i = 0; i < num_views; ++i) {
        optimized_rvecs[i].create(3, 1, CV_64F);
        optimized_tvecs[i].create(3, 1, CV_64F);
        optimized_rvecs[i].at<double>(0) = rvecs_data[i][0];
        optimized_rvecs[i].at<double>(1) = rvecs_data[i][1];
        optimized_rvecs[i].at<double>(2) = rvecs_data[i][2];
        optimized_tvecs[i].at<double>(0) = tvecs_data[i][0];
        optimized_tvecs[i].at<double>(1) = tvecs_data[i][1];
        optimized_tvecs[i].at<double>(2) = tvecs_data[i][2];
    }

    double total_squared_error = 0.0;
    int total_points = 0;

    for (int view = 0; view < num_views; ++view) {
        const int num_points = static_cast<int>(object_points[view].size());
        total_points += num_points;

        for (int point = 0; point < num_points; ++point) {
            ReprojectionErrorCeres error(
                image_points[view][point].x,
                image_points[view][point].y,
                object_points[view][point].x,
                object_points[view][point].y,
                object_points[view][point].z
            );

            double residuals[2];
            error(intrinsics, rvecs_data[view].data(), tvecs_data[view].data(), residuals);
            total_squared_error += residuals[0] * residuals[0] + residuals[1] * residuals[1];
        }
    }

    double final_rmse = std::sqrt(total_squared_error / total_points);

    BrownConradyCalibrationResult result;
    result.successfully = true;
    result.rmse = final_rmse;

    result.intrinsic = IntrinsicParams(
        optimized_camera_matrix.at<double>(0, 0),
        optimized_camera_matrix.at<double>(1, 1),
        optimized_camera_matrix.at<double>(0, 2),
        optimized_camera_matrix.at<double>(1, 2),
        optimized_camera_matrix.at<double>(0, 1)
    );

    result.distortion = std::make_unique<BrownConradyDistortion>(
        optimized_distortion.at<double>(0),
        optimized_distortion.at<double>(1),
        optimized_distortion.at<double>(2),
        optimized_distortion.at<double>(3),
        optimized_distortion.at<double>(4)
    );

    result.extrinsics_vec.clear();
    result.extrinsics_vec.reserve(num_views);
    for (int i = 0; i < num_views; ++i) {
        cv::Vec3d rvec(
            optimized_rvecs[i].at<double>(0), optimized_rvecs[i].at<double>(1), optimized_rvecs[i].at<double>(2)
        );
        cv::Vec3d tvec(
            optimized_tvecs[i].at<double>(0), optimized_tvecs[i].at<double>(1), optimized_tvecs[i].at<double>(2)
        );
        result.extrinsics_vec.emplace_back(rvec, tvec);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    result.time_seconds = std::chrono::duration<double>(end_time - start_time).count();

    return result;
}


std::unique_ptr<CalibrationResult> BrownConradyCalibrator::calibrate_impl(
    const std::vector<std::vector<cv::Point3d>> &object_points,
    const std::vector<std::vector<cv::Point2d>> &image_points
) const {
    auto result = calibrate_brown_conrady(object_points, image_points);
    return std::make_unique<BrownConradyCalibrationResult>(std::move(result));
}


BrownConradyModel BrownConradyCalibrationResult::get_camera_model(size_t external_vec_idx) const noexcept {
    if (!successfully || extrinsics_vec.empty() || external_vec_idx >= extrinsics_vec.size()) {
        return BrownConradyModel(IntrinsicParams(), ExtrinsicParams(), BrownConradyDistortion());
    }

    auto *brown_distortion = dynamic_cast<BrownConradyDistortion *>(distortion.get());
    if (!brown_distortion) {
        return BrownConradyModel(intrinsic, extrinsics_vec[external_vec_idx], BrownConradyDistortion());
    }

    return BrownConradyModel(intrinsic, extrinsics_vec[external_vec_idx], *brown_distortion);
}


}; // namespace ccl
