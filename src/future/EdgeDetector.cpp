#include "../include/EdgeDetector.hpp"


namespace ccl {


enum class EdgeOrientation : std::uint8_t
{
    NON_ORIENTATION = 0,
    HORIZONTAL_ORIENTATION,
    VERTICAL_ORIENTATION
};


const uchar ANCHOR_PIXEL = 1; // unprocessed edge pixel
const uchar EDGE_PIXEL = 255;


bool EdgeDetector::check_dir(uint8_t grad_dir, ChainDirection chain_dir) const {
    if (grad_dir == static_cast<uint8_t>(EdgeOrientation::VERTICAL_ORIENTATION) &&
        (chain_dir == ChainDirection::UP || chain_dir == ChainDirection::DOWN)) {
        return true;
    }
    if (grad_dir == static_cast<uint8_t>(EdgeOrientation::HORIZONTAL_ORIENTATION) &&
        (chain_dir == ChainDirection::LEFT || chain_dir == ChainDirection::RIGHT)) {
        return true;
    }
    return false;
};


std::pair<cv::Mat, std::vector<EdgeDetector::Segment>> EdgeDetector::detect(const cv::Mat &input_image) const {
    cv::Mat gray;
    if (input_image.channels() == 3) {
        cv::cvtColor(input_image, gray, cv::COLOR_BGR2GRAY);
    } else if (input_image.channels() == 1) {
        gray = input_image.clone();
    }

    cv::Mat blurred;
    if (params.guassian_sigma == 1.0) {
        cv::GaussianBlur(gray, blurred, cv::Size(5, 5), params.guassian_sigma);
    } else {
        cv::GaussianBlur(gray, blurred, cv::Size(), params.guassian_sigma);
    }

    auto [grad, dir] = std::move(compute_gradients_and_directions(blurred));

    auto anchor_points = std::move(find_anchor_points(grad, dir));

    cv::Mat grad_visual;
    cv::convertScaleAbs(grad, grad_visual); // Конвертируем 16SC1 в 8UC1
    cv::normalize(grad_visual, grad_visual, 0, 255, cv::NORM_MINMAX);
    cv::imwrite("debug_gradient.png", grad_visual);
    cv::imshow("Gradient Image", grad_visual);
    cv::waitKey(10); // Ненадолго показываем

    // ========== ДЕБАГ ВЫВОД 2: Карта направлений ==========
    cv::Mat dir_visual(grad.rows, grad.cols, CV_8UC1, cv::Scalar(0));
    for (int i = 0; i < grad.rows; i++) {
        for (int j = 0; j < grad.cols; j++) {
            uint8_t dir_val = dir[i * grad.cols + j];
            if (dir_val == static_cast<uint8_t>(EdgeOrientation::HORIZONTAL_ORIENTATION)) {
                dir_visual.at<uchar>(i, j) = 128; // Серый для горизонтальных
            } else if (dir_val == static_cast<uint8_t>(EdgeOrientation::VERTICAL_ORIENTATION)) {
                dir_visual.at<uchar>(i, j) = 255; // Белый для вертикальных
            }
            // 0 остается для NON_DIRECTION
        }
    }
    cv::imwrite("debug_directions.png", dir_visual);
    cv::imshow("Directions Map", dir_visual);
    cv::waitKey(10);

    // ========== ДЕБАГ ВЫВОД 3: Anchor точки ==========
    cv::Mat anchor_visual(grad.rows, grad.cols, CV_8UC1, cv::Scalar(0));
    for (const cv::Point2i &anchor : anchor_points) {
        anchor_visual.at<uchar>(anchor.y, anchor.x) = 255;
    }
    cv::imwrite("debug_anchors.png", anchor_visual);
    cv::imshow("Anchor Points", anchor_visual);
    cv::waitKey(10);

    return build_segments_and_edge_image(grad, dir, anchor_points);
}

std::pair<cv::Mat, EdgeDetector::Directions>
EdgeDetector::compute_gradients_and_directions(cv::Mat &blurred_image) const {
    int rows = blurred_image.rows;
    int cols = blurred_image.cols;
    int blur_step = blurred_image.step;

    uchar *blur_data = blurred_image.data;

    cv::Mat gradient_mat(rows, cols, CV_16SC1, cv::Scalar(0));
    short *gradient_data = (short *)gradient_mat.data;
    int gradient_step = gradient_mat.step / sizeof(short);

    std::vector<uint8_t> directions_data(rows * cols, static_cast<uint8_t>(EdgeOrientation::NON_ORIENTATION));

    for (int j = 0; j < cols; j++) {
        gradient_data[0 * gradient_step + j] = params.gradient_threshold - 1;
        gradient_data[(rows - 1) * gradient_step + j] = params.gradient_threshold - 1;
    }

    for (int i = 1; i < rows - 1; i++) {
        gradient_data[i * gradient_step + 0] = params.gradient_threshold - 1;
        gradient_data[i * gradient_step + (cols - 1)] = params.gradient_threshold - 1;
    }

    for (int i = 1; i < rows - 1; i++) {
        uchar *row_prev = blur_data + (i - 1) * blur_step;
        uchar *row_curr = blur_data + i * blur_step;
        uchar *row_next = blur_data + (i + 1) * blur_step;

        for (int j = 1; j < cols - 1; j++) {
            int gx = 0;
            int gy = 0;

            // clang-format off
            switch (params.grad_op) {
            case GradientOperator::SOBEL3_3:
                gx = abs(
                    -row_prev[j - 1] + row_prev[j + 1] +
                    -2 * row_curr[j - 1] + 2 * row_curr[j + 1] +
                    -row_next[j - 1] + row_next[j + 1]
                );
                gy = abs(
                    -row_prev[j - 1] - 2 * row_prev[j] +
                    -row_prev[j + 1] + row_next[j - 1] +
                    2 * row_next[j] + row_next[j + 1]
                );
                break;
            
            case GradientOperator::SCHARR3_3:
                gx = abs(
                    -3 * row_prev[j - 1] + 3 * row_prev[j + 1] +
                    -10 * row_curr[j - 1] + 10 * row_curr[j + 1] +
                    -3 * row_next[j - 1] + 3 * row_next[j + 1]
                );
                gy = abs(
                    -3 * row_prev[j - 1] - 10 * row_prev[j] +
                    -3 * row_prev[j + 1] + 3 * row_next[j - 1] +
                    10 * row_next[j] + 3 * row_next[j + 1]
                );
                break;
            
            case GradientOperator::PREWITT3_3:
                gx = abs(
                    -row_prev[j - 1] + row_prev[j + 1] +
                    -row_curr[j - 1] + row_curr[j + 1] +
                    -row_next[j - 1] + row_next[j + 1]
                );
                gy = abs(
                    -row_prev[j - 1] - row_prev[j] +
                    -row_prev[j + 1] + row_next[j - 1] +
                    row_next[j] + row_next[j + 1]
                );
                break;
            }
            // clang-format on

            int grad_value = 0;
            switch (params.grad_norm) {
            case GradientNorm::L1:
                grad_value = gx + gy;
                break;
            case GradientNorm::L2:
                grad_value = static_cast<int>(std::sqrt(static_cast<double>(gx * gx + gy * gy)));
                break;
            }
            gradient_data[i * gradient_step + j] = grad_value;

            if (grad_value >= params.gradient_threshold) {
                if (gx >= gy) {
                    directions_data[i * cols + j] = static_cast<uint8_t>(EdgeOrientation::VERTICAL_ORIENTATION);
                } else {
                    directions_data[i * cols + j] = static_cast<uint8_t>(EdgeOrientation::HORIZONTAL_ORIENTATION);
                }
            }
        }
    }

    return std::pair<cv::Mat, EdgeDetector::Directions>(std::move(gradient_mat), std::move(directions_data));
}


std::vector<cv::Point2i>
EdgeDetector::find_anchor_points(cv::Mat &grad_img, EdgeDetector::Directions &directions) const {
    int rows = grad_img.rows;
    int cols = grad_img.cols;
    int step = grad_img.step / sizeof(short);
    short *grad_data = (short *)grad_img.data;

    std::vector<cv::Point2i> anchor_points;

    for (int i = 2; i < rows - 2; i++) {
        int start_j = 2;
        int inc_j = 1;
        if (i % params.detail_ratio != 0) {
            start_j = params.detail_ratio;
            inc_j = params.detail_ratio;
        }

        short *row_prev = grad_data + (i - 1) * step;
        short *row_curr = grad_data + i * step;
        short *row_next = grad_data + (i + 1) * step;

        for (int j = start_j; j < cols - 2; j += inc_j) {
            if (row_curr[j] < params.gradient_threshold) {
                continue;
            }

            int dir_idx = i * cols + j;
            const auto &curr_dir = directions[dir_idx];

            if (curr_dir == static_cast<uchar>(EdgeOrientation::HORIZONTAL_ORIENTATION)) {
                if (row_curr[j] - row_prev[j] >= params.anchor_threshold &&
                    row_curr[j] - row_next[j] >= params.anchor_threshold) {
                    anchor_points.emplace_back(j, i);
                }
            } else if (curr_dir == static_cast<uchar>(EdgeOrientation::VERTICAL_ORIENTATION)) {
                if (row_curr[j] - row_curr[j - 1] >= params.anchor_threshold &&
                    row_curr[j] - row_curr[j + 1] >= params.anchor_threshold) {
                    anchor_points.emplace_back(j, i);
                }
            }
        }
    }

    return anchor_points;
}


void EdgeDetector::sort_anchor_points(cv::Mat &grad_img, std::vector<cv::Point2i> &anchor_points) const {
    if (anchor_points.empty()) {
        return;
    }

    int rows = grad_img.rows;
    int cols = grad_img.cols;
    int grad_step = grad_img.step / sizeof(short);
    short *grad_data = (short *)grad_img.data;

    short max_grad = 0;
    if (anchor_points.size() < static_cast<short>(std::numeric_limits<short>::max())) {
        for (const auto &anchor : anchor_points) {
            short grad = grad_data[anchor.y * grad_step + anchor.x];
            if (grad > max_grad) {
                max_grad = grad;
            }
        }
    } else {
        max_grad = static_cast<short>(32767);
    }

    int size = max_grad + 1;
    std::vector<int> count(size, 0);

    for (const auto &anchor : anchor_points) {
        short grad = grad_data[anchor.y * grad_step + anchor.x];
        if (grad >= 0 && grad < size) {
            count[grad]++;
        }
    }

    for (int i = 1; i < size; i++) {
        count[i] += count[i - 1];
    }

    std::vector<cv::Point2i> sorted_points(anchor_points.size());

    for (int i = anchor_points.size() - 1; i >= 0; i--) {
        const auto &anchor = anchor_points[i];
        short grad = grad_data[anchor.y * grad_step + anchor.x];

        if (grad >= 0 && grad < size) {
            int index = --count[grad];
            sorted_points[index] = anchor;
        }
    }

    std::reverse(sorted_points.begin(), sorted_points.end());
    anchor_points = std::move(sorted_points);

    std::cout << "=== ANCHOR POINTS AFTER SORTING ===" << std::endl;
    for (int i = 0; i < std::min(10, (int)anchor_points.size()); i++) {
        const auto &a = anchor_points[i];
        short grad = grad_data[a.y * grad_step + a.x];
        std::cout << "  [" << i << "] (" << a.x << "," << a.y << ") grad=" << grad << std::endl;
    }
    std::cout << "===================================" << std::endl;
}


std::pair<cv::Mat, std::vector<EdgeDetector::Segment>> EdgeDetector::build_segments_and_edge_image(
    cv::Mat &grad_img, EdgeDetector::Directions &directions, std::vector<cv::Point2i> anchor_points
) const {
    std::vector<Segment> segments;

    cv::Mat edge_img(grad_img.rows, grad_img.cols, CV_8UC1, cv::Scalar(0));
    uchar *edge_data = edge_img.data;
    int edge_step = edge_img.step;

    sort_anchor_points(grad_img, anchor_points);

    for (const auto &anchor : anchor_points) {
        edge_data[anchor.y * edge_step + anchor.x] = ANCHOR_PIXEL;
    }

    auto vec_chain_tree = process_tasks_stack(edge_img, grad_img, directions, anchor_points);

    for (auto &chain_tree : vec_chain_tree) {
        auto new_segments = std::move(build_segments_from_chain_tree(chain_tree, edge_img));
        if (!new_segments.empty()) {
            segments.insert(segments.end(), new_segments.begin(), new_segments.end());
        }
    }

    cv::Mat result = cv::Mat::zeros(grad_img.rows, grad_img.cols, CV_8UC1);
    uchar *result_data = result.data;
    int result_step = result.step;

    for (const auto &segment : segments) {
        for (const auto &pixel : segment) {
            result_data[pixel.y * result_step + pixel.x] = EDGE_PIXEL;
        }
    }

    return std::pair<cv::Mat, std::vector<Segment>>(std::move(result), std::move(segments));
}


EdgeDetector::ChainTreeVec EdgeDetector::process_tasks_stack(
    cv::Mat &edge_img, cv::Mat &grad_img, EdgeDetector::Directions &directions, std::vector<cv::Point2i> anchor_points
) const {
    std::vector<ChainTree> chain_tree_vec;

    int cols = edge_img.cols;
    int rows = edge_img.rows;
    uchar *edge_data = edge_img.data;
    int edge_step = edge_img.step;
    short *grad_data = (short *)grad_img.data;
    int grad_step = grad_img.step / sizeof(short);

    auto get_next_pixel =
        [&edge_data, &edge_step, &grad_data, &grad_step](int r, int c, ChainDirection dir) -> cv::Point2i {
        int dr = 0, dc = 0;
        int ortho1_dr = 0, ortho1_dc = 0;
        int ortho2_dr = 0, ortho2_dc = 0;

        auto edge_idx = [&edge_step](int rr, int cc) {
            return rr * edge_step + cc;
        };
        auto grad_idx = [&grad_step](int rr, int cc) {
            return rr * grad_step + cc;
        };

        edge_data[edge_idx(r, c)] = EDGE_PIXEL;

        // clang-format off
        switch (dir) {
        case ChainDirection::UP:
            dr = -1; dc = 0;
            ortho1_dr = 0; ortho1_dc = -1;
            ortho2_dr = 0; ortho2_dc = 1;
            break;
        case ChainDirection::DOWN:
            dr = 1; dc = 0;
            ortho1_dr = 0; ortho1_dc = -1;
            ortho2_dr = 0; ortho2_dc = 1;
            break;
        case ChainDirection::LEFT:
            dr = 0; dc = -1;
            ortho1_dr = -1; ortho1_dc = 0;   
            ortho2_dr = 1; ortho2_dc = 0;     
            break;
        case ChainDirection::RIGHT: 
            dr = 0; dc = 1;
            ortho1_dr = -1; ortho1_dc = 0; 
            ortho2_dr = 1; ortho2_dc = 0; 
            break;
        default:
            break;
        }
        // clang-format on

        if (edge_data[edge_idx(r + ortho1_dr, c + ortho1_dc)] == ANCHOR_PIXEL) {
            edge_data[edge_idx(r + ortho1_dr, c + ortho1_dc)] = 0;
        }
        if (edge_data[edge_idx(r + ortho2_dr, c + ortho2_dc)] == ANCHOR_PIXEL) {
            edge_data[edge_idx(r + ortho2_dr, c + ortho2_dc)] = 0;
        }

        if (edge_data[edge_idx(r + dr, c + dc)] >= ANCHOR_PIXEL) {
            return {c + dc, r + dr};
        } else if (edge_data[edge_idx(r + dr + ortho1_dr, c + dc + ortho1_dc)] >= ANCHOR_PIXEL) {
            return {c + dc + ortho1_dc, r + dr + ortho1_dr};
        } else if (edge_data[edge_idx(r + dr + ortho2_dr, c + dc + ortho2_dc)] >= ANCHOR_PIXEL) {
            return {c + dc + ortho2_dc, r + dr + ortho2_dr};
        } else {
            int g1 = grad_data[grad_idx(r + dr + ortho1_dr, c + dc + ortho1_dc)];
            int g2 = grad_data[grad_idx(r + dr, c + dc)];
            int g3 = grad_data[grad_idx(r + dr + ortho2_dr, c + dc + ortho2_dc)];

            if (g1 > g2 && g1 > g3) {
                return {c + dc + ortho1_dc, r + dr + ortho1_dr};
            } else if (g3 > g1 && g3 > g2) {
                return {c + dc + ortho2_dc, r + dr + ortho2_dr};
            } else {
                return {c + dc, r + dr};
            }
        }
    };

    for (const auto &anchor : anchor_points) {
        std::cout << "next anchor point" << std::endl;
        ChainTree chains_tree;
        auto root = chains_tree.create_node();
        Stack tasks;

        auto create_tasks = [&directions, &tasks, &cols](const cv::Point2i &p) {
            int idx = p.y * cols + p.x;
            if (directions[idx] == static_cast<uint8_t>(EdgeOrientation::VERTICAL_ORIENTATION)) {
                tasks.emplace(p.y, p.x, ChainDirection::DOWN);
                tasks.emplace(p.y, p.x, ChainDirection::UP);
            } else if (directions[idx] == static_cast<uint8_t>(EdgeOrientation::HORIZONTAL_ORIENTATION)) {
                tasks.emplace(p.y, p.x, ChainDirection::RIGHT);
                tasks.emplace(p.y, p.x, ChainDirection::LEFT);
            }
        };

        create_tasks(anchor);

        int count_pixels_in_chain_tree = 0;

        while (!tasks.empty()) {
            auto curr_task = tasks.top();
            tasks.pop();

            // if (edge_data[curr_task.i * edge_step + curr_task.j] == 0) {
            //     continue;
            // }

            // move to correct parent
            while (root->has_all_childrens()) {
                auto parent = root->get_parent();
                if (parent == nullptr) {
                    break;
                }
                if (parent->get_pixels().size() != 0) {
                    std::cout << "move to root: " << parent->get_pixels()[0] << std::endl;
                } else {
                    std::cout << "move to main root" << std::endl;
                }
                root = parent;
            }
            auto &childrens = root->get_childrens();

            std::shared_ptr<ChainNode> curr_node;
            if (edge_data[curr_task.i * edge_step + curr_task.j] != EDGE_PIXEL) {
                count_pixels_in_chain_tree++;
            }
            curr_node = chains_tree.create_node(curr_task.dir, std::vector{cv::Point2i(curr_task.j, curr_task.i)});

            int node_index = 0;
            if (curr_task.dir == ChainDirection::DOWN || curr_task.dir == ChainDirection::RIGHT) {
                node_index++;
            }

            int curr_i = curr_task.i;
            int curr_j = curr_task.j;

            std::cout << curr_j << " " << curr_i << " ";
            switch (curr_task.dir) {
            case ChainDirection::UP:
                std::cout << "UP" << " " << node_index << std::endl;
                break;
            case ChainDirection::DOWN:
                std::cout << "DOWN" << " " << node_index << std::endl;
                break;
            case ChainDirection::RIGHT:
                std::cout << "RIGHT" << " " << node_index << std::endl;
                break;
            case ChainDirection::LEFT:
                std::cout << "LEFT" << " " << node_index << std::endl;
                break;
            default:
                break;
            }
            while (true) {
                auto next_pixel = get_next_pixel(curr_i, curr_j, curr_task.dir);
                std::cout << "NEXT PIXEL" << next_pixel.x << " " << next_pixel.y << " "
                          << grad_data[next_pixel.y * grad_step + next_pixel.x] << std::endl;

                curr_i = next_pixel.y;
                curr_j = next_pixel.x;

                if (edge_data[curr_i * edge_step + curr_j] == EDGE_PIXEL ||
                    grad_data[curr_i * grad_step + curr_j] < params.gradient_threshold) {
                    std::cout << "add child: " << curr_node->get_pixels()[0];
                    if (root->get_pixels().size() != 0) {
                        std::cout << " to root: " << root->get_pixels()[0];
                    }
                    std::cout << std::endl;
                    root->add_child(curr_node, node_index);
                    break;
                }

                if (!check_dir(directions[curr_i * cols + curr_j], curr_task.dir)) {
                    std::cout << "Chain broken at next_pixel(" << curr_i << "," << curr_j << ")" << std::endl;
                    std::cout << "add child: " << curr_node->get_pixels()[0];
                    if (root->get_pixels().size() != 0) {
                        std::cout << " to root: " << root->get_pixels()[0];
                    }
                    std::cout << std::endl;
                    root->add_child(curr_node, node_index);
                    root = curr_node;
                    // create_tasks(next_pixel);
                    create_tasks(cv::Point2i(curr_j, curr_i));
                    break;
                }

                curr_node->add_pixel(std::move(next_pixel));
                count_pixels_in_chain_tree++;
            }
        }

        if (count_pixels_in_chain_tree >= params.min_path_len) {
            chain_tree_vec.emplace_back(std::move(chains_tree));
        }
    }

    return chain_tree_vec;
}


size_t EdgeDetector::find_longest_path(const std::shared_ptr<ChainNode> &node) const {
    if (!node || node->get_length() == 0) {
        return 0;
    }

    auto &childrens = node->get_childrens();

    // std::cout << "node: " << node->get_pixels()[0] << " ";
    // if (childrens[0] != nullptr) {
    //     std::cout << childrens[0]->get_pixels()[0] << " ";
    // } else {
    //     std::cout << "[]";
    // }
    // if (childrens[1] != nullptr) {
    //     std::cout << childrens[1]->get_pixels()[0] << std::endl;
    // } else {
    //     std::cout << "[]" << std::endl;
    // }

    size_t len_children_left = 0;
    if (childrens[0] != nullptr) {
        len_children_left = find_longest_path(childrens[0]);
    }

    size_t len_children_right = 0;
    if (childrens[1] != nullptr) {
        len_children_right = find_longest_path(childrens[1]);
    }

    if (len_children_left == 0 && len_children_right == 0) {
        return node->get_length();
    } else if (len_children_left > len_children_right) {
        if (childrens[1]) {
            childrens[1]->detach_from_parent();
            childrens[1] = nullptr;
        }
        return node->get_length() + len_children_left;
    } else {
        if (childrens[0]) {
            childrens[0]->detach_from_parent();
            childrens[0] = nullptr;
        }
        return node->get_length() + len_children_right;
    }
}


std::vector<EdgeDetector::Segment>
EdgeDetector::build_segments_from_chain_tree(ChainTree &chain_tree, cv::Mat &edge_img) const {
    std::vector<Segment> segments;
    auto &vec_chains = chain_tree.get_node_vec();

    auto build_segment = [](std::shared_ptr<ChainNode> &start_node) -> std::vector<cv::Point2i> {
        std::vector<cv::Point2i> pixels;
        auto &current_node = start_node;

        while (current_node) {
            const auto &node_pixels = current_node->get_pixels();
            pixels.insert(pixels.end(), node_pixels.begin(), node_pixels.end());
            current_node->clear_pixels();

            auto &children = current_node->get_childrens();
            if (children[0]) {
                current_node = children[0];
            } else if (children[1]) {
                current_node = children[1];
            } else {
                break;
            }
        }

        return pixels;
    };

    for (auto &root_chain : vec_chains) {
        auto &childrens = root_chain->get_childrens();
        size_t len_left_path = find_longest_path(childrens[0]);
        size_t len_right_path = find_longest_path(childrens[1]);
        // std::cout << "len " << len_left_path << " " << len_right_path << std::endl;
        auto pixels_left = std::move(build_segment(childrens[0]));
        auto pixels_right = std::move(build_segment(childrens[1]));

        if (len_left_path + len_right_path >= params.min_path_len) {
            std::vector<cv::Point2i> combined;
            combined.reserve(pixels_left.size() + pixels_right.size());

            combined.insert(combined.end(), pixels_right.rbegin(), pixels_right.rend());

            combined.insert(
                combined.end(), std::make_move_iterator(pixels_left.begin()), std::make_move_iterator(pixels_left.end())
            );

            segments.push_back(std::move(combined));
        } else {

            auto remove_bad_chain_tree = [](std::vector<cv::Point2i> bad_pixels, cv::Mat &edge_img) {
                uchar *edge_data = edge_img.data;
                int edge_step = edge_img.step;
                for (auto &pixel : bad_pixels) {
                    edge_data[pixel.y * edge_step + pixel.x] = 0;
                }
            };

            remove_bad_chain_tree(pixels_left, edge_img);
            remove_bad_chain_tree(pixels_right, edge_img);
        }
    }

    return segments;
}


} // namespace ccl