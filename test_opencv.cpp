// detect_opencv.cpp
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <regex>
#include <vector>

namespace fs = std::filesystem;

bool parse_filename(const std::string &filename, int &width, int &height) {
    std::regex re(R"(^(\d+)_(\d+)_)");
    std::smatch match;
    if (std::regex_search(filename, match, re)) {
        width = std::stoi(match[1].str());
        height = std::stoi(match[2].str());
        return true;
    }
    return false;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_folder> <output_folder> [--sb]\n";
        return 1;
    }

    std::string input_folder = argv[1];
    std::string output_root = argv[2];
    bool use_sb = (argc >= 4 && std::string(argv[3]) == "--sb");

    if (!fs::exists(input_folder) || !fs::is_directory(input_folder)) {
        std::cerr << "Input folder does not exist: " << input_folder << std::endl;
        return 1;
    }

    fs::create_directories(output_root);
    fs::path full_path = output_root + "/full_detected";
    fs::path not_path = output_root + "/not_detected";
    fs::create_directory(full_path);
    fs::create_directory(not_path);

    std::ofstream log_file(output_root + "/detection_log.txt");
    if (!log_file.is_open()) {
        std::cerr << "Cannot create log file.\n";
        return 1;
    }

    long long total_max_corners = 0;
    long long total_detected_corners = 0;
    int full_count = 0, not_count = 0;

    std::vector<fs::path> image_files;
    for (const auto &entry : fs::directory_iterator(input_folder)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                image_files.push_back(entry.path());
            }
        }
    }

    std::cout << "Found " << image_files.size() << " images\n";
    std::cout << "Using " << (use_sb ? "findChessboardCornersSB" : "findChessboardCorners") << "\n";

    for (const auto &img_path : image_files) {
        std::string filename = img_path.filename().string();
        int width = 0, height = 0;

        if (!parse_filename(filename, width, height)) {
            std::cerr << "Skip: cannot parse size from " << filename << "\n";
            continue;
        }

        // Исходные размеры (ширина и высота из имени)
        cv::Size pattern_size1(width, height);
        cv::Size pattern_size2(height, width);
        int max_corners = width * height; // общее количество углов одинаково

        cv::Mat img = cv::imread(img_path.string());
        if (img.empty()) {
            std::cerr << "Cannot read: " << img_path << "\n";
            continue;
        }

        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> corners;
        bool found = false;
        cv::Size used_pattern;

        // Пробуем оба варианта размеров
        if (use_sb) {
            found = cv::findChessboardCornersSB(
                gray, pattern_size1, corners, cv::CALIB_CB_EXHAUSTIVE | cv::CALIB_CB_ACCURACY
            );
            if (found) {
                used_pattern = pattern_size1;
            } else {
                found = cv::findChessboardCornersSB(
                    gray, pattern_size2, corners, cv::CALIB_CB_EXHAUSTIVE | cv::CALIB_CB_ACCURACY
                );
                if (found) {
                    used_pattern = pattern_size2;
                }
            }
        } else {
            found = cv::findChessboardCorners(
                gray, pattern_size1, corners, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE
            );
            if (found) {
                used_pattern = pattern_size1;
            } else {
                found = cv::findChessboardCorners(
                    gray, pattern_size2, corners, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE
                );
                if (found) {
                    used_pattern = pattern_size2;
                }
            }
        }

        if (found) {
            cv::cornerSubPix(
                gray,
                corners,
                cv::Size(11, 11),
                cv::Size(-1, -1),
                cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01)
            );
        }

        int detected = found ? max_corners : 0;

        fs::path dest_folder = found ? full_path : not_path;
        std::string output_path = (dest_folder / filename).string();

        cv::Mat viz = img.clone();
        if (found) {
            cv::drawChessboardCorners(viz, used_pattern, corners, found);
        }
        cv::imwrite(output_path, viz);

        log_file << img_path.string() << " -> " << output_path << "\n";
        log_file << "  corners: " << detected << " / " << max_corners << "\n";
        log_file << "  status: " << (found ? "FULL" : "NOT") << "\n\n";

        total_max_corners += max_corners;
        total_detected_corners += detected;

        if (found) {
            full_count++;
        } else {
            not_count++;
        }
    }

    double overall_percent = (total_max_corners > 0) ? 100.0 * total_detected_corners / total_max_corners : 0.0;

    std::cout << "\n=== Results ===\n";
    std::cout << "Full detected: " << full_count << "\n";
    std::cout << "Not detected: " << not_count << "\n";
    std::cout << "Corners: " << total_detected_corners << " / " << total_max_corners << " (" << std::fixed
              << std::setprecision(2) << overall_percent << "%)\n";

    log_file << "\n=== Results ===\n";
    log_file << "Full detected: " << full_count << "\n";
    log_file << "Not detected: " << not_count << "\n";
    log_file << "Corners: " << total_detected_corners << " / " << total_max_corners << " (" << std::fixed
             << std::setprecision(2) << overall_percent << "%)\n";

    log_file.close();
    return 0;
}
