#pragma once
#include <math.h>
#include <vector>


namespace ccl {


class NFA {
private:
    std::vector<int> LUT;
    double prob;
    double logNT;

public:
    NFA() = delete;
    NFA(double prob, double logNT);
    ~NFA() = default;

    bool check_validation(int n, int k);
    static double atan2(double y, double x);

private:
    double nfa(int n, int k);
    static double log_gamma_lanczos(double x);
    static double log_gamma_windschitl(double x) noexcept;
    static double log_gamma(double x) noexcept;
    static bool double_equal(double a, double b) noexcept;
};


}; // namespace ccl
