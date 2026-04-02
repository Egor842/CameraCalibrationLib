#include "../include/CameraModel.hpp"


namespace ccl {


IntrinsicParams::IntrinsicParams(double fx, double fy, double cx, double cy, double skew)
    : fx(fx),
      fy(fy),
      cx(cx),
      cy(cy),
      skew(skew) {}


cv::Mat IntrinsicParams::to_cv_mat() const {
    return (cv::Mat_<double>(3, 3) << fx, skew, cx, 0, fy, cy, 0, 0, 1);
}


IntrinsicParams::operator cv::Mat() const {
    return to_cv_mat();
}


bool IntrinsicParams::from_cv_mat(const cv::Mat &k) {
    if (k.rows != 3 && k.cols != 3) {
        return false;
    }
    fx = k.at<double>(0, 0);
    fy = k.at<double>(1, 1);
    cx = k.at<double>(0, 2);
    cy = k.at<double>(1, 2);
    skew = k.at<double>(0, 1);
    return true;
}


bool IntrinsicParams::is_valid() const {
    return fx > 0 && fy > 0 && cx >= 0 && cy >= 0;
}


double IntrinsicParams::fx_error(const IntrinsicParams &other) const noexcept {
    return std::abs(fx - other.fx);
}


double IntrinsicParams::fy_error(const IntrinsicParams &other) const noexcept {
    return std::abs(fy - other.fy);
}


double IntrinsicParams::cx_error(const IntrinsicParams &other) const noexcept {
    return std::abs(cx - other.cx);
}


double IntrinsicParams::cy_error(const IntrinsicParams &other) const noexcept {
    return std::abs(cy - other.cy);
}


double IntrinsicParams::skew_error(const IntrinsicParams &other) const noexcept {
    return std::abs(skew - other.skew);
}


ExtrinsicParams::ExtrinsicParams() : rvec(0, 0, 0), tvec(0, 0, 0) {}


ExtrinsicParams::ExtrinsicParams(const cv::Vec3d &r, const cv::Vec3d &t) : rvec(r), tvec(t) {}


ExtrinsicParams::ExtrinsicParams(cv::Vec3d &&r, cv::Vec3d &&t) : rvec(std::move(r)), tvec(std::move(t)) {}


cv::Vec3d ExtrinsicParams::get_rvec() const {
    return rvec;
}


cv::Vec3d ExtrinsicParams::get_tvec() const {
    return tvec;
}


cv::Mat ExtrinsicParams::rvec_mat() const {
    return cv::Mat(3, 1, CV_64F, (void *)rvec.val).clone();
}


cv::Mat ExtrinsicParams::tvec_mat() const {
    return cv::Mat(3, 1, CV_64F, (void *)tvec.val).clone();
}


cv::Matx33d ExtrinsicParams::rotation_matrix() const {
    cv::Matx33d R;
    cv::Rodrigues(rvec_mat(), R);
    return R;
}


ExtrinsicParams ExtrinsicParams::compose(const ExtrinsicParams &other) const {
    cv::Matx33d R1 = rotation_matrix();
    cv::Matx33d R2 = other.rotation_matrix();
    cv::Matx33d R_combined = R1 * R2;
    cv::Vec3d t_combined = tvec + cv::Vec3d(R1 * other.tvec);
    cv::Vec3d r_combined;
    cv::Rodrigues(cv::Mat(R_combined), r_combined);
    return ExtrinsicParams(r_combined, t_combined);
}


ExtrinsicParams ExtrinsicParams::inverse() const {
    cv::Matx33d R_inv = rotation_matrix().t();
    cv::Vec3d t_inv = -cv::Vec3d(R_inv * tvec);
    cv::Vec3d r_inv;
    cv::Rodrigues(cv::Mat(R_inv), r_inv);
    return ExtrinsicParams(r_inv, t_inv);
}


bool ExtrinsicParams::is_valid() const {
    return !std::isnan(rvec[0]) && !std::isnan(rvec[1]) && !std::isnan(rvec[2]) && !std::isnan(tvec[0]) &&
           !std::isnan(tvec[1]) && !std::isnan(tvec[2]);
}


std::optional<ExtrinsicParams> ExtrinsicParams::create_from_mat(const cv::Mat &R, const cv::Mat &t) {
    if (R.empty() || R.rows != 3 || R.cols != 3 || R.type() != CV_64F) {
        return std::nullopt;
    }
    if (t.empty() || t.rows != 3 || t.cols != 1 || t.type() != CV_64F) {
        return std::nullopt;
    }

    cv::Mat rvec_mat;
    try {
        cv::Rodrigues(R, rvec_mat);
    } catch (...) { return std::nullopt; }

    if (rvec_mat.empty() || rvec_mat.rows != 3 || rvec_mat.cols != 1) {
        return std::nullopt;
    }

    auto rvec = cv::Vec3d(rvec_mat.at<double>(0, 0), rvec_mat.at<double>(1, 0), rvec_mat.at<double>(2, 0));
    auto tvec = cv::Vec3d(t.at<double>(0, 0), t.at<double>(1, 0), t.at<double>(2, 0));

    return ExtrinsicParams(std::move(rvec), std::move(tvec));
}


double ExtrinsicParams::rvec_error(const ExtrinsicParams &other) const noexcept {
    return cv::norm(rvec - other.rvec);
}


double ExtrinsicParams::tvec_error(const ExtrinsicParams &other) const noexcept {
    return cv::norm(tvec - other.tvec);
}


CameraModel::CameraModel(
    const IntrinsicParams &intrisic, const ExtrinsicParams &external, std::unique_ptr<DistortionModel> dist
)
    : intrisic(intrisic),
      external(external),
      distortion(std::move(dist)) {}


IntrinsicParams CameraModel::get_intrinsic() const noexcept {
    return intrisic;
}


ExtrinsicParams CameraModel::get_external() const noexcept {
    return external;
}


} // namespace ccl
