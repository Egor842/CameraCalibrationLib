#pragma once
#include "EdgeDetectorParamsFree.hpp"


namespace ccl {


// Ellipse Equation:
// Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
struct EllipseEquation {
    static constexpr size_t COEFF_SIZE = 7;
    std::array<double, COEFF_SIZE> coeff = {0.0};

    EllipseEquation(double new_coeff[7]) {
        for (size_t idx = 0; idx < COEFF_SIZE; idx++) {
            coeff[idx] = new_coeff[idx];
        }
    }
    EllipseEquation() = default;

    double A() {
        return coeff[1];
    }
    double B() {
        return coeff[2];
    }
    double C() {
        return coeff[3];
    }
    double D() {
        return coeff[4];
    }
    double E() {
        return coeff[5];
    }
    double F() {
        return coeff[6];
    }
};


struct Circle {
    cv::Point2d center;
    double radius;
    std::vector<cv::Point2i> pixels;
    EllipseEquation equation;
    double RMSE;
    bool is_ellipse;
};


class CircleDetector {

public:
    using Circles = std::vector<Circle>;

private:
    Circles circles;

public:
    CircleDetector() = default;
    Circles detect(const cv::Mat &img);
};


}; // namespace ccl
