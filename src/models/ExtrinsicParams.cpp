#include "../../include/ccl/models/ExtrinsicParams.hpp"


namespace ccl {


cv::Mat ExtrinsicParams::rvec_mat() const {
    return cv::Mat(rvec, true);
}


cv::Mat ExtrinsicParams::tvec_mat() const {
    return cv::Mat(tvec, true);
}


cv::Matx33d ExtrinsicParams::rotation_matrix() const {
    cv::Matx33d R;
    cv::Rodrigues(rvec, R);
    return R;
}


cv::Matx44d ExtrinsicParams::to_homogeneous() const {
    cv::Matx44d T = cv::Matx44d::eye();
    cv::Matx33d R = rotation_matrix();

    T(0, 0) = R(0, 0);
    T(0, 1) = R(0, 1);
    T(0, 2) = R(0, 2);
    T(1, 0) = R(1, 0);
    T(1, 1) = R(1, 1);
    T(1, 2) = R(1, 2);
    T(2, 0) = R(2, 0);
    T(2, 1) = R(2, 1);
    T(2, 2) = R(2, 2);

    T(0, 3) = tvec[0];
    T(1, 3) = tvec[1];
    T(2, 3) = tvec[2];

    return T;
}


ExtrinsicParams ExtrinsicParams::compose(const ExtrinsicParams &other) const {
    cv::Matx33d R1 = rotation_matrix();
    cv::Matx33d R2 = other.rotation_matrix();

    cv::Matx33d R_new = R1 * R2;
    cv::Vec3d t2 = other.tvec;
    cv::Vec3d t_new = R1 * t2 + tvec;

    cv::Vec3d r_new;
    cv::Rodrigues(R_new, r_new);

    return ExtrinsicParams(r_new, t_new);
}


ExtrinsicParams ExtrinsicParams::inverse() const {
    cv::Matx33d R = rotation_matrix();
    cv::Matx33d R_inv = R.t();

    cv::Vec3d t_inv = -R_inv * tvec;
    cv::Vec3d r_inv;
    cv::Rodrigues(R_inv, r_inv);

    return ExtrinsicParams(r_inv, t_inv);
}


bool ExtrinsicParams::is_valid() const {
    for (int i = 0; i < 3; ++i) {
        if (!std::isfinite(rvec[i]) || !std::isfinite(tvec[i])) {
            return false;
        }
    }

    double norm_r = cv::norm(rvec);
    if (!std::isfinite(norm_r)) {
        return false;
    }

    return true;
}


bool ExtrinsicParams::from_cv_mat(const cv::Mat &R, const cv::Mat &t) {
    if (R.rows != 3 || R.cols != 3) {
        return false;
    }

    if (t.rows != 3 || t.cols != 1) {
        return false;
    }

    if (R.depth() != CV_64F || t.depth() != CV_64F) {
        return false;
    }

    cv::Rodrigues(R, rvec);

    tvec[0] = t.at<double>(0);
    tvec[1] = t.at<double>(1);
    tvec[2] = t.at<double>(2);

    return true;
}


}; // namespace ccl
