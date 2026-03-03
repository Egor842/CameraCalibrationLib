#include "include/ChessboardDetector.hpp"
#include "include/EdgeDetector.hpp"
#include "include/EdgeDetectorParamsFree.hpp"
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <thread>


// int main() {
//     cv::Mat image = cv::imread("/home/egor/CameraCalibrationLib/build/real_test.jpg");
//     if (image.empty()) {
//         std::cerr << "Failed to load image!" << std::endl;
//         return -1;
//     }

//     try {
//         ccl::EdgeDetectorParams params;
//         params.gradient_threshold = 15;
//         params.anchor_threshold = 3;
//         params.grad_norm = ccl::GradientNorm::L1;
//         params.grad_op = ccl::GradientOperator::PREWITT3_3;
//         params.detail_ratio = 1;
//         params.min_path_len = 10;
//         // params.gaussian_kernel = cv::Size(3, 3);
//         params.guassian_sigma = 1.0;

//         // ccl::EdgeDetector detector(params);
//         ccl::EdgeDetectorParamsFree detector;

//         auto start = std::chrono::steady_clock::now();
//         auto [result_img, segments] = detector.detect(image);

//         auto end = std::chrono::steady_clock::now();
//         auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

//         std::cout << "Time taken: " << duration.count() << " ms" << std::endl;

//         cv::Mat visualization = cv::Mat::zeros(image.size(), CV_8UC3);
//         cv::imwrite("res.png", result_img);
//         std::cout << "Found " << segments.size() << " segments" << std::endl;

//         std::vector<cv::Scalar> colors = {
//             cv::Scalar(255, 0, 0),     // синий
//             cv::Scalar(0, 255, 0),     // зелёный
//             cv::Scalar(0, 0, 255),     // красный
//             cv::Scalar(255, 255, 0),   // голубой
//             cv::Scalar(255, 0, 255),   // фиолетовый
//             cv::Scalar(0, 255, 255),   // жёлтый
//             cv::Scalar(128, 128, 255), // розовый
//             cv::Scalar(128, 255, 128), // светло-зелёный
//             cv::Scalar(255, 128, 128), // светло-красный
//             cv::Scalar(192, 192, 192)  // серый
//         };

//         for (size_t i = 0; i < segments.size(); i++) {
//             const auto &segment = segments[i];
//             cv::Scalar color = colors[i % colors.size()];

//             for (const auto &point : segment) {
//                 if (point.x >= 0 && point.x < visualization.cols && point.y >= 0 && point.y < visualization.rows) {
//                     visualization.at<cv::Vec3b>(point.y, point.x) = cv::Vec3b(color[0], color[1], color[2]);
//                 }
//             }

//             std::cout << "Drawing segment " << i + 1 << "/" << segments.size() << " (" << segment.size() << "points)"
//                       << std::endl;

//             cv::imshow("Edge Detection Progress", visualization);

//             int key = cv::waitKey(2); 
//             if (key == 27) {          
//                 std::cout << "Interrupted by user" << std::endl;
//                 break;
//             }
//         }

//         cv::imshow("Final Result", visualization);
//         cv::imwrite("segments_visualization.png", visualization);

//         cv::Mat overlay = image.clone();
//         for (size_t i = 0; i < segments.size(); i++) {
//             const auto &segment = segments[i];
//             cv::Scalar color = colors[i % colors.size()];

//             for (const auto &point : segment) {
//                 if (point.x >= 0 && point.x < overlay.cols && point.y >= 0 && point.y < overlay.rows) {
//                     cv::circle(overlay, point, 0, color, -1);
//                 }
//             }
//         }

//         cv::imshow("Overlay on Original", overlay);
//         cv::imwrite("overlay.png", overlay);

//         std::cout << "Press any key to exit..." << std::endl;
//         cv::waitKey(0);

//     } catch (const std::exception &e) {
//         std::cerr << "Error: " << e.what() << '\n';
//         return -1;
//     }

//     return 0;
// }


int main() {
    cv::Mat image = cv::imread("/home/egor/CameraCalibrationLib/data/test1.png");
    // cv::Mat image = cv::imread("/home/egor/CameraCalibrationLib/data/e5.png");
    if (image.empty()) {
        std::cerr << "Failed to load image!" << std::endl;
        return -1;
    }

    ccl::ChessboardDetectorParams params;
    // params.scales = {0.5, 1.0, 2.0};
    params.handle_occlusions = true;
    ccl::ChessboardDetector detector(params);

    auto boards = detector.detect(image);

    if (boards.empty()) {
        std::cout << "No chessboards found!" << std::endl;
        return 0;
    }

    std::cout << "Found " << boards.size() << " chessboard(s)" << std::endl;

    cv::Mat display_image = image.clone();

    for (size_t board_idx = 0; board_idx < boards.size(); ++board_idx) {
        const auto &board = boards[board_idx];
        const auto &board_corners = board.pixels;

        for (const auto &pt : board_corners) {
            if (pt.x >= 0 && pt.y >= 0) {
                cv::circle(display_image, pt, 5, cv::Scalar(0, 0, 255), -1);
            }
        }
        cv::imshow("Detected Chessboards", display_image);
        cv::waitKey(1000);
    }

    cv::imshow("Detected Chessboards", display_image);
    cv::imwrite("detected_boards.png", display_image);
    cv::waitKey(0);

    cv::destroyAllWindows();
    return 0;
}
