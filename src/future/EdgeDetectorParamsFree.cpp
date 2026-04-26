#include "../include/EdgeDetectorParamsFree.hpp"


namespace ccl {


const double EPSILON = 1.0;
const uint8_t EDGE_PIXEL = 255;


std::pair<cv::Mat, std::vector<EdgeDetectorParamsFree::Segment>>
EdgeDetectorParamsFree::detect(const cv::Mat &input_image) const {
    cv::Mat gray;
    cv::cvtColor(input_image, gray, cv::COLOR_BGR2GRAY);

    cv::Mat blurred;
    // cv::GaussianBlur(gray, blurred, params.gaussian_kernel, params.guassian_sigma);
    cv::GaussianBlur(gray, blurred, cv::Size(), 1.0);

    auto [grad, dir] = std::move(EdgeDetector::compute_gradients_and_directions(blurred));

    auto anchor_points = std::move(EdgeDetector::find_anchor_points(grad, dir));

    auto [edge_image, raw_segments] = std::move(EdgeDetector::build_segments_and_edge_image(grad, dir, anchor_points));

    compute_H(grad);

    Np = 0;
    for (const auto &segment : raw_segments) {
        int len = segment.size();
        // len = (len + 1) / 2;
        Np += (len * (len - 1)) / 2;
        // Np += len;
    }

    edge_image.setTo(0);
    for (auto &segment : raw_segments) {
        test_segment(edge_image, grad, segment, 0, segment.size() - 1);
    }

    // extract_new_segments();

    cv::imshow("GG", edge_image);
    cv::waitKey(0);

    return {edge_image, raw_segments};
}


void EdgeDetectorParamsFree::compute_H(const cv::Mat &grad_img) const {
    const int MAX_GRAD_VALUE = 32768; // 128*256!

    short *grad_data = (short *)grad_img.data;
    int rows = grad_img.rows;
    int cols = grad_img.cols;
    int grad_step = grad_img.step / sizeof(short);

    std::vector<int> hist(MAX_GRAD_VALUE, 0);
    int size = (rows - 2) * (cols - 2);

    for (int y = 1; y < rows - 1; y++) {
        for (int x = 1; x < cols - 1; x++) {
            short grad = grad_data[y * grad_step + x];
            // if (grad < params.gradient_threshold) {
            //     size--;
            // }
            hist[grad]++;
        }
    }

    for (int i = MAX_GRAD_VALUE - 1; i > 0; i--) {
        hist[i - 1] += hist[i];
    }

    H.assign(MAX_GRAD_VALUE, 0.0);
    for (int i = 0; i < MAX_GRAD_VALUE; i++) {
        H[i] = static_cast<double>(hist[i]) / static_cast<double>(size);
    }

    std::cout << "H[0] = " << H[0] << std::endl;
    std::cout << "H[25] = " << H[25] << std::endl;
    std::cout << "H[429] = " << H[429] << std::endl;
}


void EdgeDetectorParamsFree::test_segment(
    cv::Mat &edge_image, const cv::Mat &grad_img, Segment &segment, int first_index, int second_index
) const {
    int len = second_index - first_index + 1;
    if (len < params.min_path_len) {
        return;
    }

    short *grad_data = (short *)grad_img.data;
    int rows = grad_img.rows;
    int cols = grad_img.cols;
    int grad_step = grad_img.step / sizeof(short);

    short min_grad = std::numeric_limits<short>::max();
    int min_grad_index;
    for (int idx = first_index; idx <= second_index; idx++) {
        const auto &pixel = segment[idx];
        if (grad_data[pixel.y * grad_step + pixel.x] < min_grad) {
            min_grad = grad_data[pixel.y * grad_step + pixel.x];
            min_grad_index = idx;
        }
    }

    double nfa = calculate_nfa(H[min_grad], (len) / 2);

    if (nfa <= EPSILON) {
        for (int idx = first_index; idx <= second_index; idx++) {
            const auto &pixel = segment[idx];
            edge_image.at<uchar>(pixel.y, pixel.x) = EDGE_PIXEL;
        }

        return;
    }

    int end = min_grad_index - 1;
    while (end > first_index) {
        const auto &pixel = segment[end];
        if (grad_data[pixel.y * grad_step + pixel.x] <= min_grad) {
            end--;
        } else {
            break;
        }
    }

    int start = min_grad_index + 1;
    while (start < second_index) {
        const auto &pixel = segment[start];
        if (grad_data[pixel.y * grad_step + pixel.x] <= min_grad) {
            start++;
        } else {
            break;
        }
    }

    test_segment(edge_image, grad_img, segment, first_index, end);
    test_segment(edge_image, grad_img, segment, start, second_index);
}


double EdgeDetectorParamsFree::calculate_nfa(double H_value, int len) const {
    double nfa = Np;
    for (int idx = 0; idx < len && nfa > EPSILON; idx++) {
        nfa *= H_value;
    }

    return nfa;
}


}; // namespace ccl