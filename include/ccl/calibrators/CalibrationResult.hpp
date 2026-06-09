#pragma once
#include "../models/Distortion.hpp"
#include "../models/ExtrinsicParams.hpp"
#include "../models/IntrinsicParams.hpp"
#include <type_traits>


namespace ccl {


template <typename DistortionType> struct CalibrationResult {
    static_assert(std::is_base_of_v<Distortion, DistortionType>, "DistortionType must inherit from ccl::Distortion");

    IntrinsicParams intrinsic{};
    DistortionType distortion{};
    std::vector<ExtrinsicParams> extrinsics_vec{};
    std::vector<double> per_view_errors{};
    double rmse = 0.0;
    double time_seconds = 0.0;
    bool successfully = false;

    void save_calibration_results_xml(const std::string &filename, cv::Size image_size = {-1, -1}) const;
};


template <typename DistortionType>
void CalibrationResult<DistortionType>::save_calibration_results_xml(
    const std::string &filename, cv::Size image_size
) const {
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        std::cerr << "Failed to open " << filename << " for writing\n";
        return;
    }

    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::string time_str = std::ctime(&now_c);
    time_str.pop_back();
    fs << "calibration_time" << time_str;
    fs << "camera_poses" << static_cast<int>(extrinsics_vec.size());
    fs << "image_width" << image_size.width;
    fs << "image_height" << image_size.height;

    fs << "work_time" << time_seconds;

    fs << "camera_matrix" << intrinsic.to_cv_mat();
    const auto &coeffs = distortion.get_coefficients();
    fs << "distortion_coefficients" << "[";
    for (size_t i = 0; i < coeffs.size(); ++i) {
        if (i) {
            fs << ", ";
        }
        fs << coeffs[i];
    }
    fs << "]";
    fs << "reprojection_RMS_error" << rmse;
    fs << "Max_reprojection_RMS_error" << *std::max_element(per_view_errors.begin(), per_view_errors.end());

    fs << "per_view_reprojection_errors" << "[";
    for (size_t i = 0; i < per_view_errors.size(); ++i) {
        fs << "{:" << "file" << ("frame_" + std::to_string(i) + ".yml") << "err" << per_view_errors[i] << "}";
    }
    fs << "]";

    cv::Mat extrinsics(static_cast<int>(extrinsics_vec.size()), 6, CV_64F);
    for (size_t i = 0; i < extrinsics_vec.size(); ++i) {
        extrinsics.at<double>(i, 0) = extrinsics_vec[i].rvec[0];
        extrinsics.at<double>(i, 1) = extrinsics_vec[i].rvec[1];
        extrinsics.at<double>(i, 2) = extrinsics_vec[i].rvec[2];
        extrinsics.at<double>(i, 3) = extrinsics_vec[i].tvec[0];
        extrinsics.at<double>(i, 4) = extrinsics_vec[i].tvec[1];
        extrinsics.at<double>(i, 5) = extrinsics_vec[i].tvec[2];
    }
    fs << "extrinsic_parameters" << extrinsics;

    fs.release();
}


}; // namespace ccl
