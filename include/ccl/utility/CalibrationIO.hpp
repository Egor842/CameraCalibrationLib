#pragma once
#include "../calibrators/CalibrationResult.hpp"
#include "../detectors/PatternSize.hpp"
#include <algorithm>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <memory>
#include <numeric>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/persistence.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>


namespace ccl {


inline void save_calibration_results_xml(
    const std::string &filename,
    const cv::Mat &K,
    const cv::Mat &dist,
    double rms,
    const std::vector<double> &per_view_errors,
    const std::vector<cv::Mat> &rvecs,
    const std::vector<cv::Mat> &tvecs,
    int nframes,
    int pattern_w,
    int pattern_h,
    double step_size,
    cv::Size image_size,
    double work_time
) {
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

    fs << "nframes" << nframes;
    fs << "image_width" << image_size.width;
    fs << "image_height" << image_size.height;
    fs << "pattern_width" << pattern_w;
    fs << "pattern_height" << pattern_h;
    fs << "step_size" << step_size;
    fs << "work_time" << work_time;

    fs << "camera_matrix" << K;
    fs << "distortion_coefficients" << dist;
    fs << "reprojection_RMS_error" << rms;
    fs << "Max_reprojection_RMS_error" << *std::max_element(per_view_errors.begin(), per_view_errors.end());

    fs << "per_view_reprojection_errors" << "[";
    for (size_t i = 0; i < per_view_errors.size(); ++i) {
        fs << "{:" << "file" << ("frame_" + std::to_string(i) + ".yml") << "err" << per_view_errors[i] << "}";
    }
    fs << "]";

    cv::Mat extrinsics(static_cast<int>(rvecs.size()), 6, CV_64F);
    for (size_t i = 0; i < rvecs.size(); ++i) {
        extrinsics.at<double>(i, 0) = rvecs[i].at<double>(0);
        extrinsics.at<double>(i, 1) = rvecs[i].at<double>(1);
        extrinsics.at<double>(i, 2) = rvecs[i].at<double>(2);
        extrinsics.at<double>(i, 3) = tvecs[i].at<double>(0);
        extrinsics.at<double>(i, 4) = tvecs[i].at<double>(1);
        extrinsics.at<double>(i, 5) = tvecs[i].at<double>(2);
    }
    fs << "extrinsic_parameters" << extrinsics;

    fs.release();
}


}; // namespace ccl
