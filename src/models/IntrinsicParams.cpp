#include "../../include/ccl/models/IntrinsicParams.hpp"


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


bool IntrinsicParams::from_cv_mat(const cv::Mat &k) noexcept {
    if (k.rows != 3 || k.cols != 3 || k.depth() != CV_64F) {
        return false;
    }
    fx = k.at<double>(0, 0);
    fy = k.at<double>(1, 1);
    cx = k.at<double>(0, 2);
    cy = k.at<double>(1, 2);
    skew = k.at<double>(0, 1);
    return true;
}


bool IntrinsicParams::is_valid(const cv::Size &img_size) const noexcept {
    if (fx <= 0.0 || fy <= 0.0) {
        return false;
    }

    if (img_size.width > 0 && img_size.height > 0) {
        if (cx < -img_size.width || cx > img_size.width) {
            return false;
        }
        if (cy < -img_size.height || cy > img_size.height) {
            return false;
        }
    }

    return true;
}


} // namespace ccl
