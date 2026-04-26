#include "../include/ChainTree.hpp"


namespace ccl {


ChainNode::ChainNode(ChainDirection dir, const std::vector<cv::Point> &pixels)
    : direction(dir),
      pixels(pixels),
      parent() {
    childrens.fill(nullptr);
}

ChainDirection ChainNode::get_direction() const {
    return direction;
}


const std::vector<cv::Point2i> &ChainNode::get_pixels() const {
    return pixels;
}


std::shared_ptr<ChainNode> ChainNode::get_parent() const {
    return parent.lock();
}


const std::array<std::shared_ptr<ChainNode>, 2> &ChainNode::get_childrens() const {
    return childrens;
}


std::array<std::shared_ptr<ChainNode>, 2> &ChainNode::get_childrens() {
    return childrens;
}


size_t ChainNode::get_length() const {
    return pixels.size();
}


void ChainNode::add_pixel(const cv::Point &pixel) {
    pixels.push_back(pixel);
}


void ChainNode::add_pixels(const std::vector<cv::Point> &pixels) {
    this->pixels.insert(pixels.end(), pixels.begin(), pixels.end());
}


void ChainNode::clear_pixels() {
    pixels.clear();
}


bool ChainNode::add_child(std::shared_ptr<ChainNode> child, int index) {
    if (child == nullptr || (index != 0 && index != 1)) {
        return false;
    }

    child->parent = shared_from_this();
    childrens[index] = child;
    return true;
}


bool ChainNode::detach_from_parent() {
    if (auto p = parent.lock()) {
        auto &parent_children = p->get_childrens();
        for (int i = 0; i < 2; i++) {
            if (parent_children[i].get() == this) {
                parent_children[i] = nullptr;
                parent.reset();
                return true;
            }
        }
    }
    return false;
}


bool ChainNode::has_all_childrens() const {
    return childrens[0] != nullptr && childrens[1] != nullptr;
}


bool ChainNode::is_root() const {
    return parent.expired();
}


std::shared_ptr<ChainNode> ChainTree::create_node(ChainDirection dir, const std::vector<cv::Point> &pixels) {
    auto root = std::make_shared<ChainNode>(dir, pixels);
    nodes.push_back(root);
    return root;
}


// DFS
void ChainTree::traverse(const std::shared_ptr<ChainNode> &node, std::function<void(const ChainNode &)> visit) const {
    if (!node) {
        return;
    }

    visit(*node);

    for (const auto &child : node->get_childrens()) {
        traverse(child, visit);
    }
}


const std::vector<std::shared_ptr<ChainNode>> &ChainTree::get_node_vec() const {
    return nodes;
}


const std::shared_ptr<ChainNode> &ChainTree::get_root() const {
    return nodes[0];
}


} // namespace ccl