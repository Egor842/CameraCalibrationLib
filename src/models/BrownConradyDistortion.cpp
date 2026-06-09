#include "../../include/ccl/models/BrownConradyDistortion.hpp"
#include <fstream>
#include <memory>
#include <yaml-cpp/yaml.h>


namespace ccl {


constexpr size_t MAX_ITER = 10;
constexpr double EPSILON = 1e-6;


BrownConradyDistortion::BrownConradyDistortion(double k1_, double k2_, double p1_, double p2_, double k3_)
    : k1(k1_),
      k2(k2_),
      p1(p1_),
      p2(p2_),
      k3(k3_) {}


void BrownConradyDistortion::distort(double x, double y, double &xd, double &yd) const noexcept {
    double x2 = x * x;
    double y2 = y * y;
    double r2 = x2 + y2;
    double r4 = r2 * r2;
    double r6 = r4 * r2;

    double radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
    double tx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x2);
    double ty = p1 * (r2 + 2.0 * y2) + 2.0 * p2 * x * y;

    xd = x * radial + tx;
    yd = y * radial + ty;
}


void BrownConradyDistortion::undistort(double xd, double yd, double &x, double &y) const noexcept {
    x = xd;
    y = yd;

    for (size_t iter = 0; iter < MAX_ITER; ++iter) {
        double x2 = x * x;
        double y2 = y * y;
        double r2 = x2 + y2;
        double r4 = r2 * r2;
        double r6 = r4 * r2;

        double radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
        double d_radial_dr2 = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4;

        double tx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x2);
        double ty = p1 * (r2 + 2.0 * y2) + 2.0 * p2 * x * y;

        double dtx_dx = 2.0 * p1 * y + p2 * (2.0 * x + 4.0 * x);
        double dtx_dy = 2.0 * p1 * x + p2 * (2.0 * y);
        double dty_dx = p1 * (2.0 * y) + 2.0 * p2 * y;
        double dty_dy = p1 * (2.0 * y + 4.0 * y) + 2.0 * p2 * x;

        double J11 = radial + x * x * d_radial_dr2 * 2.0 + dtx_dx;
        double J12 = x * y * d_radial_dr2 * 2.0 + dtx_dy;
        double J21 = x * y * d_radial_dr2 * 2.0 + dty_dx;
        double J22 = radial + y * y * d_radial_dr2 * 2.0 + dty_dy;

        double det = J11 * J22 - J12 * J21;
        if (std::abs(det) < 1e-12) {
            break;
        }

        double fx_val = x * radial + tx - xd;
        double fy_val = y * radial + ty - yd;

        double dx = (J22 * fx_val - J12 * fy_val) / det;
        double dy = (J11 * fy_val - J21 * fx_val) / det;

        x -= dx;
        y -= dy;

        if (std::abs(dx) < EPSILON && std::abs(dy) < EPSILON) {
            break;
        }
    }
}


std::vector<double> BrownConradyDistortion::get_coefficients() const noexcept {
    return {k1, k2, p1, p2, k3};
}


bool BrownConradyDistortion::set_coefficients(const std::vector<double> &coeffs) {
    if (coeffs.size() < 5) {
        return false;
    }
    k1 = coeffs[0];
    k2 = coeffs[1];
    p1 = coeffs[2];
    p2 = coeffs[3];
    k3 = coeffs[4];
    return true;
}


} // namespace ccl
