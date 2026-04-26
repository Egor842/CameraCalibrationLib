#include "../../include/ccl/models/CameraModel.hpp"


namespace ccl {


bool CameraModel::set_pose(const ExtrinsicParams &pose) {
    if (pose.is_valid()) {
        this->pose = pose;
        return true;
    }

    return false;
}


bool CameraModel::set_intrinsic(const IntrinsicParams &K, const cv::Size &img_size) {
    if (K.is_valid(img_size)) {
        this->K = K;
        return true;
    }

    return false;
};


IntrinsicParams CameraModel::get_intrins() const noexcept {
    return K;
};


ExtrinsicParams CameraModel::get_pose() const noexcept {
    return pose;
};


}; // namespace ccl
