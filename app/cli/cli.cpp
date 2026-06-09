#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <sstream>
#include <filesystem>
#include <optional>
#include <cctype>

#include <opencv2/opencv.hpp>

#include "ccl/calibrators/ZhangCalibrator.hpp"
#include "ccl/detectors/ChessboardDetector.hpp"
#include "ccl/detectors/Pattern.hpp"
#include "ccl/calibrators/CalibrationResult.hpp"
#include "ccl/models/BrownConradyDistortion.hpp"
#include "ccl/detectors/PatternSize.hpp"
#include "ccl/utility/DetectionIO.hpp"

struct CalibrationConfig {
    std::string images_dir;
    int pattern_width = 9;
    int pattern_height = 6;
    double square_size = 0.025;
    std::string output_xml = "calibration_result.xml";
    bool estimate_k3 = false;
    bool estimate_skew = false;
    ccl::LossFunctionType loss_function = ccl::LossFunctionType::CAUCHY;
    std::string detector_config_path;
    bool save_detections = true;

    ccl::PatternSize patternSize() const {
        return ccl::PatternSize(pattern_width, pattern_height);
    }

    std::string toCommandLine(const std::string& exe_name) const {
        std::ostringstream cmd;
        cmd << exe_name;
        cmd << " --images_dir \"" << images_dir << "\"";
        cmd << " --pattern_width " << pattern_width;
        cmd << " --pattern_height " << pattern_height;
        cmd << " --square_size " << square_size;
        cmd << " --output_xml \"" << output_xml << "\"";
        if (estimate_k3) cmd << " --k3";
        if (estimate_skew) cmd << " --skew";
        cmd << " --loss ";
        switch (loss_function) {
            case ccl::LossFunctionType::L2:    cmd << "L2";    break;
            case ccl::LossFunctionType::HUBER: cmd << "HUBER"; break;
            case ccl::LossFunctionType::CAUCHY:cmd << "CAUCHY";break;
            case ccl::LossFunctionType::TUKEY: cmd << "TUKEY"; break;
        }
        if (!detector_config_path.empty())
            cmd << " --detector_config \"" << detector_config_path << "\"";
        if (!save_detections)
            cmd << " --no-detections";
        return cmd.str();
    }
};

std::vector<cv::Point3d> generateObjectPoints(const ccl::PatternSize& size, double squareSize) {
    std::vector<cv::Point3d> pts;
    int w = static_cast<int>(size.get_width());
    int h = static_cast<int>(size.get_height());
    pts.reserve(w * h);
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            pts.emplace_back(j * squareSize, i * squareSize, 0.0);
        }
    }
    return pts;
}

ccl::ChessboardDetector createDetector(const std::string& configPath) {
    if (!configPath.empty()) {
        std::cout << "Detector config loading not implemented, using default parameters.\n";
    }
    return ccl::ChessboardDetector();
}

bool processImages(const std::string& dir,
                   const ccl::PatternSize& boardSize,
                   double squareSize,
                   ccl::ChessboardDetector& detector,
                   std::vector<std::vector<cv::Point2d>>& imagePoints,
                   std::vector<std::vector<cv::Point3d>>& objectPoints,
                   std::vector<ccl::Chessboard>& usedBoards,
                   std::vector<std::string>& usedImagePaths,
                   cv::Size& imageSize) {
    namespace fs = std::filesystem;
    const std::vector<std::string> exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"};
    std::vector<fs::path> allPaths;

    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (std::find(exts.begin(), exts.end(), ext) != exts.end()) {
                allPaths.push_back(entry.path());
            }
        }
    }

    if (allPaths.empty()) {
        std::cerr << "No images found in directory.\n";
        return false;
    }
    std::sort(allPaths.begin(), allPaths.end());

    const std::vector<cv::Point3d> objPts = generateObjectPoints(boardSize, squareSize);
    const size_t totalPoints = objPts.size();
    const size_t totalImages = allPaths.size();
    size_t processedCount = 0;
    size_t usedFrames = 0;

    std::cout << "Found " << totalImages << " images. Starting detection...\n";

    for (size_t idx = 0; idx < totalImages; ++idx) {
        const auto& path = allPaths[idx];
        processedCount++;

        std::cout << "[" << processedCount << "/" << totalImages << "] "
                  << path.filename().string() << " ... ";

        cv::Mat img = cv::imread(path.string(), cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::cout << "FAIL (cannot load)" << std::endl;
            continue;
        }
        if (imageSize.width <= 0) {
            imageSize = img.size();
        }

        auto boards = detector.detect(img);
        bool used = false;
        for (auto& board : boards) {
            if (board.get_size().get_width() != boardSize.get_width() ||
                board.get_size().get_height() != boardSize.get_height()) {
                continue;
            }

            auto matches = board.get_valid_matches(objPts);
            if (matches.image_points.empty()) {
                continue;
            }

            if (matches.image_points.size() < 4) {
                std::cout << "FAIL (only " << matches.image_points.size() << " points, need >=4)" << std::endl;
                continue;
            }

            imagePoints.push_back(matches.image_points);
            objectPoints.push_back(matches.object_points);
            usedBoards.push_back(board);
            usedImagePaths.push_back(path.string());
            usedFrames++;
            used = true;

            std::cout << "OK (" << matches.image_points.size() << "/" << totalPoints << " points)" << std::endl;
            break;
        }
        if (!used) {
            std::cout << "FAIL (no suitable board)" << std::endl;
        }
    }

    std::cout << "Processed " << processedCount << " images, "
              << usedFrames << " usable views found (partial or full).\n";

    return !imagePoints.empty();
}

void printUsage(const std::string& exeName) {
    std::cout << "Usage: " << exeName << " [OPTIONS]\n"
              << "Options:\n"
              << "  --images_dir <path>        (required) Directory with calibration images\n"
              << "  --pattern_width <int>      Number of internal corners horizontally (default 9)\n"
              << "  --pattern_height <int>     Number of internal corners vertically (default 6)\n"
              << "  --square_size <double>     Size of a square in meters (default 0.025)\n"
              << "  --output_xml <path>        Output XML file (default calibration_result.xml)\n"
              << "  --loss <L2|HUBER|CAUCHY|TUKEY>  Loss function (default CAUCHY)\n"
              << "  --k3                       Estimate k3 distortion coefficient\n"
              << "  --skew                     Estimate skew coefficient\n"
              << "  --detector_config <path>   YAML config for chessboard detector (optional)\n"
              << "  --no-detections            Do not save detection visualizations and YAML\n"
              << "\nIf no arguments are given, the program runs interactively.\n";
}

bool parseArguments(int argc, char* argv[], CalibrationConfig& cfg) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--images_dir") {
            if (i + 1 < argc) cfg.images_dir = argv[++i];
            else return false;
        } else if (arg == "--pattern_width") {
            if (i + 1 < argc) cfg.pattern_width = std::stoi(argv[++i]);
            else return false;
        } else if (arg == "--pattern_height") {
            if (i + 1 < argc) cfg.pattern_height = std::stoi(argv[++i]);
            else return false;
        } else if (arg == "--square_size") {
            if (i + 1 < argc) cfg.square_size = std::stod(argv[++i]);
            else return false;
        } else if (arg == "--output_xml") {
            if (i + 1 < argc) cfg.output_xml = argv[++i];
            else return false;
        } else if (arg == "--loss") {
            if (i + 1 < argc) {
                std::string loss = argv[++i];
                if (loss == "L2") cfg.loss_function = ccl::LossFunctionType::L2;
                else if (loss == "HUBER") cfg.loss_function = ccl::LossFunctionType::HUBER;
                else if (loss == "CAUCHY") cfg.loss_function = ccl::LossFunctionType::CAUCHY;
                else if (loss == "TUKEY") cfg.loss_function = ccl::LossFunctionType::TUKEY;
                else return false;
            } else return false;
        } else if (arg == "--k3") {
            cfg.estimate_k3 = true;
        } else if (arg == "--skew") {
            cfg.estimate_skew = true;
        } else if (arg == "--detector_config") {
            if (i + 1 < argc) cfg.detector_config_path = argv[++i];
            else return false;
        } else if (arg == "--no-detections") {
            cfg.save_detections = false;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return false;
        }
    }
    if (cfg.images_dir.empty()) {
        std::cerr << "Missing required --images_dir\n";
        return false;
    }
    ccl::PatternSize ps(cfg.pattern_width, cfg.pattern_height);
    cfg.pattern_width = static_cast<int>(ps.get_width());
    cfg.pattern_height = static_cast<int>(ps.get_height());
    return true;
}

bool interactiveConfig(CalibrationConfig& cfg) {
    std::cout << "Enter images directory: ";
    std::getline(std::cin, cfg.images_dir);
    if (cfg.images_dir.empty()) return false;

    std::cout << "Enter pattern width (internal corners): ";
    std::cin >> cfg.pattern_width;
    std::cout << "Enter pattern height (internal corners): ";
    std::cin >> cfg.pattern_height;
    std::cout << "Enter square size (m): ";
    std::cin >> cfg.square_size;
    std::cin.ignore();

    ccl::PatternSize ps(cfg.pattern_width, cfg.pattern_height);
    cfg.pattern_width = static_cast<int>(ps.get_width());
    cfg.pattern_height = static_cast<int>(ps.get_height());

    std::cout << "Enter output XML file path [calibration_result.xml]: ";
    std::string out;
    std::getline(std::cin, out);
    if (!out.empty()) cfg.output_xml = out;

    char ch;
    std::cout << "Estimate k3? (y/n) [n]: ";
    std::cin >> ch;
    cfg.estimate_k3 = (ch == 'y' || ch == 'Y');

    std::cout << "Estimate skew? (y/n) [n]: ";
    std::cin >> ch;
    cfg.estimate_skew = (ch == 'y' || ch == 'Y');

    int lossChoice = 3;
    std::cout << "Choose loss function:\n1 - L2\n2 - HUBER\n3 - CAUCHY\n4 - TUKEY\nEnter choice [3]: ";
    std::cin >> lossChoice;
    switch (lossChoice) {
        case 1: cfg.loss_function = ccl::LossFunctionType::L2; break;
        case 2: cfg.loss_function = ccl::LossFunctionType::HUBER; break;
        case 3: cfg.loss_function = ccl::LossFunctionType::CAUCHY; break;
        case 4: cfg.loss_function = ccl::LossFunctionType::TUKEY; break;
        default: cfg.loss_function = ccl::LossFunctionType::CAUCHY;
    }
    std::cin.ignore();

    std::cout << "Enter path to detector config YAML (leave empty for default): ";
    std::getline(std::cin, cfg.detector_config_path);

    std::cout << "Save detection visualizations? (y/n) [y]: ";
    std::cin >> ch;
    cfg.save_detections = (ch != 'n' && ch != 'N');
    std::cin.ignore();

    return true;
}

int main(int argc, char* argv[]) {
    CalibrationConfig config;
    std::string exe_name = argv[0];

    if (argc > 1) {
        if (!parseArguments(argc, argv, config)) {
            printUsage(exe_name);
            return 1;
        }
    } else {
        std::cout << "=== Interactive Camera Calibration ===\n";
        if (!interactiveConfig(config)) {
            return 1;
        }
    }

    auto detector = createDetector(config.detector_config_path);

    std::vector<std::vector<cv::Point2d>> imagePoints;
    std::vector<std::vector<cv::Point3d>> objectPoints;
    std::vector<ccl::Chessboard> usedBoards;
    std::vector<std::string> usedImagePaths;
    cv::Size imageSize;

    if (!processImages(config.images_dir,
                       config.patternSize(),
                       config.square_size,
                       detector,
                       imagePoints,
                       objectPoints,
                       usedBoards,
                       usedImagePaths,
                       imageSize)) {
        std::cerr << "No usable chessboard views found. Exiting.\n";
        return 1;
    }

    std::cout << "Successfully detected " << imagePoints.size() << " chessboard views.\n";

    ccl::ZhangCalibrator calibrator;
    if(config.estimate_k3) {
        calibrator.set_estimation_k3();   
    }
    if(config.estimate_skew) {
        calibrator.set_estimation_skew();   
    }
    calibrator.set_loss_function(config.loss_function);

    auto result = calibrator.calibrate(objectPoints, imagePoints, imageSize);

    std::cout << "\n================ Calibration Results ================\n";
    std::cout << "RMS reprojection error: " << result.rmse << "\n";
    std::cout << "Camera matrix:\n" << result.intrinsic.to_cv_mat() << "\n";
    const auto& distCoeffs = result.distortion.get_coefficients();
    std::cout << "Distortion coefficients (k1,k2,p1,p2,k3): ";
    for (size_t i = 0; i < distCoeffs.size(); ++i) {
        std::cout << distCoeffs[i] << " ";
    }
    std::cout << "\nWork time: " << result.time_seconds << " seconds\n";

    if (!config.output_xml.empty()) {
        result.save_calibration_results_xml(config.output_xml, imageSize);
        std::cout << "Calibration saved to " << config.output_xml << "\n";
    }

    if (config.save_detections && !config.output_xml.empty() && !usedBoards.empty()) {
        std::filesystem::path xmlPath(config.output_xml);
        std::filesystem::path detectionDir = xmlPath.parent_path() / "detections";
        std::filesystem::create_directories(detectionDir);

        for (size_t i = 0; i < usedBoards.size(); ++i) {
            cv::Mat imgColor = cv::imread(usedImagePaths[i], cv::IMREAD_COLOR);
            if (imgColor.empty()) {
                std::cerr << "Failed to load image for visualization: " << usedImagePaths[i] << "\n";
                continue;
            }

            usedBoards[i].vizualize(imgColor);

            std::string imgFilename = "detection_" + std::to_string(i) + ".jpg";
            cv::imwrite((detectionDir / imgFilename).string(), imgColor);

            std::string yamlFilename = "detection_" + std::to_string(i) + ".yml";
            ccl::utils::save_detection_data_yaml(
                (detectionDir / yamlFilename).string(),
                usedBoards[i],
                objectPoints[i],
                config.square_size,
                imageSize
            );

            std::cout << "Saved detection #" << i << " to " << detectionDir << "\n";
        }
    }

    std::cout << "\nTo repeat this calibration non-interactively, run:\n";
    std::cout << config.toCommandLine(exe_name) << std::endl;

    return 0;
}
