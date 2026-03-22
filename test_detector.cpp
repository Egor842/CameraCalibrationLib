#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <regex>
#include <utility>
#include <vector>

#include "include/ChessboardDetector.hpp"

namespace fs = std::filesystem;

struct ImageInfo {
    std::string original_path;
    std::string output_path;
    int max_corners;
    int detected_corners;
    std::vector<std::string> pattern;
};

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

std::vector<std::string> make_pattern(const std::vector<std::optional<cv::Point2d>> &pixels, int width, int height) {
    std::vector<std::string> rows;
    for (int r = 0; r < height; ++r) {
        std::string row;
        for (int c = 0; c < width; ++c) {
            int idx = r * width + c;
            if (idx < (int)pixels.size() && pixels[idx].has_value()) {
                row += "* ";
            } else {
                row += "x ";
            }
        }
        if (!row.empty()) {
            row.pop_back();
        }
        rows.push_back(row);
    }
    return rows;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_folder> <output_folder>\n";
        return 1;
    }

    std::string input_folder = argv[1];
    std::string output_root = argv[2];

    if (!fs::exists(input_folder) || !fs::is_directory(input_folder)) {
        std::cerr << "Input folder does not exist: " << input_folder << std::endl;
        return 1;
    }

    fs::create_directories(output_root);
    fs::path full_detected_path = output_root + "/full_detected";
    fs::path partial_path = output_root + "/partical_detected";
    fs::path not_detected_path = output_root + "/not_detected";
    fs::create_directory(full_detected_path);
    fs::create_directory(partial_path);
    fs::create_directory(not_detected_path);

    fs::path partial_low = partial_path / "0_35";
    fs::path partial_mid = partial_path / "35_70";
    fs::path partial_high = partial_path / "70_100";
    fs::create_directory(partial_low);
    fs::create_directory(partial_mid);
    fs::create_directory(partial_high);

    std::ofstream log_file(output_root + "/detection_log.txt");
    if (!log_file.is_open()) {
        std::cerr << "Cannot create log file.\n";
        return 1;
    }

    long long total_max_corners = 0;
    long long total_detected_corners = 0;

    ccl::ChessboardDetector detector;

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

    std::cout << "Found " << image_files.size() << " image files.\n";

    for (const auto &img_path : image_files) {
        std::string filename = img_path.filename().string();
        int width = 0, height = 0;
        if (!parse_filename(filename, width, height)) {
            std::cerr << "Warning: cannot parse size from filename: " << filename << ". Skipping.\n";
            continue;
        }

        if (width < height) {
            std::swap(width, height);
        }
        int max_corners = width * height;
        cv::Mat img = cv::imread(img_path.string());
        if (img.empty()) {
            std::cerr << "Cannot read image: " << img_path << std::endl;
            continue;
        }

        auto boards = detector.detect(img);

        ImageInfo info;
        info.original_path = img_path.string();
        info.max_corners = max_corners;

        const ccl::Chessboard *matched_board = nullptr;
        for (const auto &board : boards) {
            const auto &board_size = board.get_size();
            int board_width = board_size.get_width();
            int board_height = board_size.get_height();
            if ((board_width == width && board_height == height) || (board_width == height && board_height == width)) {
                matched_board = &board;
                break;
            }
        }

        if (matched_board == nullptr) {
            info.detected_corners = 0;
            info.pattern = make_pattern({}, width, height);
            info.output_path = (not_detected_path / filename).string();
            cv::imwrite(info.output_path, img);
        } else {
            const auto &pixels = matched_board->get_pixels_vec();
            int detected = std::count_if(pixels.begin(), pixels.end(), [](const auto &opt) {
                return opt.has_value();
            });

            info.detected_corners = detected;
            info.pattern = make_pattern(pixels, width, height);

            double percent = (detected * 100.0) / max_corners;
            fs::path dest_folder;
            if (detected == max_corners) {
                dest_folder = full_detected_path;
            } else {
                if (percent < 35.0) {
                    dest_folder = partial_low;
                } else if (percent < 70.0) {
                    dest_folder = partial_mid;
                } else {
                    dest_folder = partial_high;
                }
            }
            info.output_path = (dest_folder / filename).string();

            cv::Mat viz = img.clone();
            viz = matched_board->vizualize(viz);
            cv::imwrite(info.output_path, viz);
        }

        log_file << "Image: " << info.original_path << "\n";
        log_file << "Output: " << info.output_path << "\n";
        log_file << "Max corners: " << info.max_corners << ", Detected: " << info.detected_corners << "\n";
        log_file << "Pattern:\n";
        for (const auto &row : info.pattern) {
            log_file << row << "\n";
        }
        log_file << "\n";

        total_max_corners += info.max_corners;
        total_detected_corners += info.detected_corners;
    }

    double overall_percent = (total_max_corners > 0) ? (100.0 * total_detected_corners / total_max_corners) : 0.0;
    std::cout << "\n=== Overall Statistics ===\n";
    std::cout << "Total max corners: " << total_max_corners << "\n";
    std::cout << "Total detected corners: " << total_detected_corners << "\n";
    std::cout << "Detection rate: " << std::fixed << std::setprecision(2) << overall_percent << "%\n";

    log_file << "\n=== Overall Statistics ===\n";
    log_file << "Total max corners: " << total_max_corners << "\n";
    log_file << "Total detected corners: " << total_detected_corners << "\n";
    log_file << "Detection rate: " << std::fixed << std::setprecision(2) << overall_percent << "%\n";

    log_file.close();
    return 0;
}
