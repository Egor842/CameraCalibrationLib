#include "include/BrownConradyCalibrator.hpp"
#include "include/BrownConradyModel.hpp"
#include "include/ChessboardDetector.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <optional>

namespace fs = std::filesystem;

struct DetectionResult {
    std::string path;
    std::optional<ccl::Chessboard> board;
    bool ok;
};

std::vector<std::string> get_images(const std::string &dir) {
    std::vector<std::string> images;
    for (const auto &entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        std::string ext = entry.path().extension().string();
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
            images.push_back(entry.path().string());
        }
    }
    return images;
}

DetectionResult process_image(const std::string &path, const ccl::PatternSize &pattern) {
    DetectionResult res;
    res.path = path;

    cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        res.ok = false;
        return res;
    }

    ccl::ChessboardDetectorParams params;
    params.scales = {0.5, 1.0};
    ccl::ChessboardDetector detector(params);
    auto boards = detector.detect(img);

    if (boards.empty()) {
        res.ok = false;
        return res;
    }

    res.board = boards[0];
    res.ok = true;
    return res;
}

std::pair<std::vector<cv::Point3d>, std::vector<cv::Point2d>>
extract_points(const ccl::Chessboard &board, double square_size) {
    std::vector<cv::Point3d> obj;
    for (int i = 0; i < board.get_size().get_height(); ++i) {
        for (int j = 0; j < board.get_size().get_width(); ++j) {
            obj.emplace_back(j * square_size, i * square_size, 0.0);
        }
    }

    auto correct_points = board.matches_points(obj);
    return {correct_points.second, correct_points.first};
}

void save_board_image(const std::string &src, const std::string &dst, const ccl::Chessboard &board) {
    fs::create_directories(fs::path(dst).parent_path());

    cv::Mat img = cv::imread(src);
    cv::Mat viz = board.vizualize(img);
    cv::imwrite(dst, viz);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <image_dir> [pattern_width pattern_height]" << std::endl;
        return 1;
    }

    std::string input_dir = argv[1];
    int pattern_w = (argc > 2) ? std::stoi(argv[2]) : 9;
    int pattern_h = (argc > 3) ? std::stoi(argv[3]) : 6;

    ccl::PatternSize pattern(pattern_w, pattern_h);
    double square_size = 0.025;

    std::string good_dir = input_dir + "/good";
    std::string partial_dir = input_dir + "/partial";
    std::string bad_dir = input_dir + "/bad";

    fs::create_directories(good_dir);
    fs::create_directories(partial_dir);
    fs::create_directories(bad_dir);

    auto images = get_images(input_dir);
    std::cout << "Found " << images.size() << " images\n";

    std::vector<DetectionResult> good, partial, bad;

    for (const auto &path : images) {
        auto res = process_image(path, pattern);

        if (!res.ok) {
            bad.push_back(res);
            std::cout << "[BAD] " << fs::path(path).filename() << std::endl;
            continue;
        }

        int expected = pattern.value().area();
        int detected = res.board->get_size().value().area();

        if (detected > expected) {
            bad.push_back(res);
            save_board_image(path, bad_dir + "/" + fs::path(path).filename().string(), res.board.value());
            std::cout << "[BAD] " << fs::path(path).filename() << " (size mismatch)" << std::endl;
        } else if (detected == expected && res.board->full_detected) {
            good.push_back(res);
            save_board_image(path, good_dir + "/" + fs::path(path).filename().string(), res.board.value());
            std::cout << "[GOOD] " << fs::path(path).filename() << std::endl;
        } else {
            partial.push_back(res);
            save_board_image(path, partial_dir + "/" + fs::path(path).filename().string(), res.board.value());
            std::cout << "[PARTIAL] " << fs::path(path).filename() << std::endl;
        }
    }

    std::cout << "\nGood: " << good.size() << " Partial: " << partial.size() << " Bad: " << bad.size() << std::endl;

    if (good.size() + partial.size() < 3) {
        std::cerr << "Not enough boards for calibration" << std::endl;
        return 1;
    }

    std::vector<std::vector<cv::Point3d>> all_obj;
    std::vector<std::vector<cv::Point2d>> all_img;

    for (const auto &r : good) {
        auto [obj, img] = extract_points(r.board.value(), square_size);
        all_obj.push_back(obj);
        all_img.push_back(img);
    }

    for (const auto &r : partial) {
        auto [obj, img] = extract_points(r.board.value(), square_size);
        all_obj.push_back(obj);
        all_img.push_back(img);
    }

    cv::Mat first_img = cv::imread(good.empty() ? partial[0].path : good[0].path);
    cv::Size image_size = first_img.size();

    auto start = std::chrono::high_resolution_clock::now();
    ccl::BrownConradyCalibrator calibrator(image_size);
    auto result = calibrator.calibrate_brown_conrady(all_obj, all_img);
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    if (result.successfully) {
        std::cout << "\nCalibration successful!" << std::endl;
        std::cout << "RMS: " << result.rmse << " px" << std::endl;
        std::cout << "Time: " << elapsed << " sec" << std::endl;

        auto model = result.get_camera_model();
        std::cout << "fx=" << model.get_intrinsic().fx << " fy=" << model.get_intrinsic().fy << std::endl;
        std::cout << "cx=" << model.get_intrinsic().cx << " cy=" << model.get_intrinsic().cy << std::endl;

        cv::FileStorage fs("calibration_result.yaml", cv::FileStorage::WRITE);
        fs << "fx" << model.get_intrinsic().fx;
        fs << "fy" << model.get_intrinsic().fy;
        fs << "cx" << model.get_intrinsic().cx;
        fs << "cy" << model.get_intrinsic().cy;
        fs << "skew" << model.get_intrinsic().skew;

        auto dist = model.get_distortion();
        auto *brown = dynamic_cast<ccl::BrownConradyDistortion *>(dist.get());
        if (brown) {
            auto coeffs = brown->get_coefficients();
            fs << "k1" << coeffs[0];
            fs << "k2" << coeffs[1];
            fs << "p1" << coeffs[2];
            fs << "p2" << coeffs[3];
            fs << "k3" << coeffs[4];
        }

        fs << "rmse" << result.rmse;
        fs << "time_seconds" << elapsed;
        fs.release();

        std::cout << "\nResults saved to calibration_result.yaml" << std::endl;
    } else {
        std::cerr << "Calibration failed" << std::endl;
    }

    return 0;
}
