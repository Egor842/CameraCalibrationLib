#pragma once
#include "Distortion.hpp"
#include <vector>


namespace ccl {


class BrownConradyDistortion : public Distortion {
private:
    double k1 = 0.0, k2 = 0.0;
    double p1 = 0.0, p2 = 0.0;
    double k3 = 0.0;

public:
    BrownConradyDistortion() = default;
    BrownConradyDistortion(double k1_, double k2_, double p1_, double p2_, double k3_ = 0.0);
    BrownConradyDistortion(const BrownConradyDistortion &) = default;
    BrownConradyDistortion(BrownConradyDistortion &&) = default;
    ~BrownConradyDistortion() = default;

    BrownConradyDistortion &operator=(const BrownConradyDistortion &) = default;
    BrownConradyDistortion &operator=(BrownConradyDistortion &&) = default;

    void distort(double x, double y, double &xd, double &yd) const noexcept override;
    void undistort(double xd, double yd, double &x, double &y) const noexcept override;
    [[nodiscard]] std::vector<double> get_coefficients() const noexcept override;
    [[nodiscard]] bool set_coefficients(const std::vector<double> &coeffs) override;
};


} // namespace ccl
