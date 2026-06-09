/**
 * @file NFA.cpp
 * @brief Implementation of Number of False Alarms (NFA) criterion
 *
 * ============================================================================
 * NFA (Number of False Alarms) - statistical criterion for detecting
 * structures in images, proposed by Desolneux, Moisan, Morel (2000).
 * ============================================================================
 *
 * MAIN IDEA:
 * ----------
 * We test whether an observation (k matches out of n points) is random
 * or meaningful. To do this, we compute the probability of getting THE SAME OR BETTER
 * observation by chance:
 *
 *   P = P(X ≥ k) = Σ_{i=k}^n C(n,i) * p^i * (1-p)^(n-i)
 *
 * Where p is the probability of random matching (usually 1/8 = 0.125 for 8 directions)
 *
 * Then we compare with the total number of tests (NT):
 *
 *   NFA = NT * P
 *
 * If NFA < 1 - the result is meaningful (less than one false alarm per image)
 *
 * In the code we work with logarithms: log10(NFA) = log10(NT) + log10(P)
 *
 *
 * NUMERICAL COMPUTATION:
 * ----------------------
 * Direct computation is impossible due to huge numbers (factorials). We use:
 *
 * 1. Logarithms for the first term (i=k):
 *      log(P(k)) = log(C(n,k)) + k*log(p) + (n-k)*log(1-p)
 *      where log(C(n,k)) = log(n!) - log(k!) - log((n-k)!)
 *
 * 2. Recurrence relation for subsequent terms:
 *      term[i] = term[i-1] * (n-i+1)/i * p/(1-p)
 *
 * 3. Stop when error estimate < 10%
 *
 *
 * WHERE TO FIND DETAILS:
 * ---------------------
 * Original paper:
 *   Desolneux, A., Moisan, L., & Morel, J. M. (2000).
 *   "Meaningful alignments"
 *
 * Numerical implementation based on LSD (Line Segment Detector) code:
 *   http://www.ipol.im/pub/art/2012/gjmr-lsd/
 *
 * Approximation details:
 *   - Lanczos approximation: sin(πx) formula, accurate for small x
 *   - Windschitl approximation: fast for large x (>15)
 *
 *
 * CLASS STRUCTURE:
 * ----------------
 * - NFALUT          : builds lookup table for fast validation
 * - nfa()           : main NFA computation for given n,k
 * - log_gamma_*()   : logarithm of gamma function (factorial for any number)
 * - myAtan2()       : fast arctangent with LUT
 * - double_equal()  : double comparison with tolerance
 */


#include "../include/NFA.hpp"
#include <float.h>


namespace ccl {


const size_t INV_CACHE_SIZE = 100000;
const size_t MAX_LUT_SIZE = 1024;
const double RELATIVE_ERROR_FACTOR = 100.0;


NFA::NFA(double prob, double logNT) : prob(prob), logNT(logNT) {
    LUT.push_back(1);
    int k = 1;
    for (int n = 1; n < MAX_LUT_SIZE * 10; n++) {
        bool found_valid = false;
        double nfa_value;

        while (k <= n) {
            nfa_value = nfa(n, k);

            if (nfa_value >= 0) {
                found_valid = true;
                break;
            }

            k++;
        }

        if (found_valid) {
            LUT.push_back(k);
        } else {
            LUT.push_back(MAX_LUT_SIZE * 10 + 1);
        }
    }
}


bool NFA::check_validation(int n, int k) {
    if (n >= LUT.size()) {
        return nfa(n, k) >= 0.0;
    } else {
        return k >= LUT[n];
    }
}


double NFA::atan2(double y, double x) {
    static double LUT[MAX_LUT_SIZE + 1];
    static bool tableInited = false;
    if (!tableInited) {
        for (int i = 0; i <= MAX_LUT_SIZE; i++) {
            LUT[i] = atan((double)i / MAX_LUT_SIZE);
        }

        tableInited = true;
    }

    double abs_y = fabs(y);
    double abs_x = fabs(x);

    bool invert = false;
    if (abs_y > abs_x) {
        double t = abs_x;
        abs_x = abs_y;
        abs_y = t;
        invert = true;
    }

    double ratio;
    if (abs_x == 0) {
        abs_x = 1e-6;
    }

    ratio = abs_y / abs_x;

    double angle = LUT[(int)(ratio * MAX_LUT_SIZE)];

    if (x >= 0) {
        if (y >= 0) {
            // 1 quadrant
            if (invert) {
                angle = M_PI / 2 - angle;
            }

        } else {
            // 4 quadrant
            if (invert == false) {
                angle = M_PI - angle;
            } else {
                angle = M_PI / 2 + angle;
            }
        }

    } else {
        if (y >= 0) {
            /// 2 quadrant
            if (invert == false) {
                angle = M_PI - angle;
            } else {
                angle = M_PI / 2 + angle;
            }

        } else {
            /// 3 quadrant
            if (invert) {
                angle = M_PI / 2 - angle;
            }
        }
    }

    return angle;
}


double NFA::nfa(int total_points, int aligned_points) {
    static double inv_cache[INV_CACHE_SIZE];
    double tolerance = 0.1;
    double log_first_term, current_term, binom_ratio, mult_factor, tail_sum, trunc_error, p_ratio;
    int i;

    if (total_points < 0 || aligned_points < 0 || aligned_points > total_points || prob <= 0.0 || prob >= 1.0) {
        return -1.0;
    }

    if (total_points == 0 || aligned_points == 0) {
        return -logNT;
    }
    if (total_points == aligned_points) {
        return -logNT - (double)total_points * log10(prob);
    }

    p_ratio = prob / (1.0 - prob);

    log_first_term = log_gamma((double)total_points + 1.0) - log_gamma((double)aligned_points + 1.0) -
                     log_gamma((double)(total_points - aligned_points) + 1.0) + (double)aligned_points * log(prob) +
                     (double)(total_points - aligned_points) * log(1.0 - prob);
    current_term = exp(log_first_term);

    if (double_equal(current_term, 0.0)) {
        if ((double)aligned_points > (double)total_points * prob) {
            return -log_first_term / M_LN10 - logNT;
        } else {
            return -logNT;
        }
    }

    tail_sum = current_term;

    for (i = aligned_points + 1; i <= total_points; i++) {
        double inv_i;
        if (i < INV_CACHE_SIZE) {
            if (inv_cache[i] != 0.0) {
                inv_i = inv_cache[i];
            } else {
                inv_i = 1.0 / (double)i;
                inv_cache[i] = inv_i;
            }
        } else {
            inv_i = 1.0 / (double)i;
        }

        binom_ratio = (double)(total_points - i + 1) * inv_i;
        mult_factor = binom_ratio * p_ratio;

        current_term *= mult_factor;
        tail_sum += current_term;

        if (binom_ratio < 1.0) {
            int remaining_terms = total_points - i + 1;
            trunc_error =
                current_term * ((1.0 - pow(mult_factor, (double)remaining_terms)) / (1.0 - mult_factor) - 1.0);

            double current_nfa = -log10(tail_sum) - logNT;
            double max_allowed_error = tolerance * fabs(current_nfa) * tail_sum;

            if (trunc_error < max_allowed_error) {
                break;
            }
        }
    }

    return -log10(tail_sum) - logNT;
}

double NFA::log_gamma_lanczos(double x) {
    static double LANCZOS_COEFFS[7] = {
        75122.6331530, 80916.6278952, 36308.2951477, 8687.24529705, 1168.92649479, 83.8676043424, 2.50662827511
    };

    double base_term = (x + 0.5) * log(x + 5.5) - (x + 5.5);
    double series_sum = 0.0;

    for (int idx = 0; idx < 7; idx++) {
        base_term -= log(x + (double)idx);
        series_sum += LANCZOS_COEFFS[idx] * pow(x, (double)idx);
    }

    return base_term + log(series_sum);
}


double NFA::log_gamma_windschitl(double x) noexcept {
    return 0.918938533204673 + (x - 0.5) * log(x) - x + 0.5 * x * log(x * sinh(1 / x) + 1 / (810.0 * pow(x, 6.0)));
}


double NFA::log_gamma(double x) noexcept {
    return (x > 15) ? log_gamma_windschitl(x) : log_gamma_lanczos(x);
}


bool NFA::double_equal(double a, double b) noexcept {
    if (a == b) {
        return true;
    }

    double abs_diff = fabs(a - b);
    double abs_max = std::max(fabs(a), fabs(b));

    abs_max = std::max(abs_max, std::numeric_limits<double>::min());

    return (abs_diff / abs_max) <= (RELATIVE_ERROR_FACTOR * std::numeric_limits<double>::epsilon());
}


}; // namespace ccl