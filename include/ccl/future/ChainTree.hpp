#pragma once
#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>


namespace ccl {


enum class ChainDirection : uint8_t
{
    UP = 0, // 0
    DOWN,   // 1
    LEFT,   // 2
    RIGHT,  // 3
    NON
};


class ChainNode : public std::enable_shared_from_this<ChainNode> {
private:
    ChainDirection direction;
    std::vector<cv::Point2i> pixels;
    std::weak_ptr<ChainNode> parent;
    std::array<std::shared_ptr<ChainNode>, 2> childrens;

public:
    ChainNode(ChainDirection dir = ChainDirection::NON, const std::vector<cv::Point> &pixels = {});
    ChainNode(const ChainNode &) = default;
    ChainNode(ChainNode &&) = default;
    ~ChainNode() = default;

    ChainDirection get_direction() const;
    const std::vector<cv::Point2i> &get_pixels() const;
    std::shared_ptr<ChainNode> get_parent() const;
    const std::array<std::shared_ptr<ChainNode>, 2> &get_childrens() const;
    std::array<std::shared_ptr<ChainNode>, 2> &get_childrens();
    size_t get_length() const;

    void add_pixel(const cv::Point &pixel);
    void add_pixels(const std::vector<cv::Point> &pixels);
    void clear_pixels();

    bool add_child(std::shared_ptr<ChainNode> child, int index);
    bool detach_from_parent();

    bool has_all_childrens() const;
    bool is_root() const;
};


class ChainTree {
private:
    std::vector<std::shared_ptr<ChainNode>> nodes;

public:
    ChainTree() = default;
    ChainTree(const ChainTree &) = default;
    ChainTree(ChainTree &&) = default;
    ~ChainTree() = default;

    std::shared_ptr<ChainNode>
    create_node(ChainDirection dir = ChainDirection::NON, const std::vector<cv::Point> &pixels = {});

    void traverse(const std::shared_ptr<ChainNode> &node, std::function<void(const ChainNode &)> visit) const;

    const std::vector<std::shared_ptr<ChainNode>> &get_node_vec() const;

    const std::shared_ptr<ChainNode> &get_root() const;
};


} // namespace ccl
