#pragma once
#include "../models/Distortion.hpp"
#include "../models/ExtrinsicParams.hpp"
#include "../models/IntrinsicParams.hpp"
#include <type_traits>


namespace ccl {


template <typename DistortionType> struct CalibrationResult {
    static_assert(std::is_base_of_v<Distortion, DistortionType>, "DistortionType must inherit from ccl::Distortion");

    IntrinsicParams intrinsic{};
    DistortionType distortion{};
    std::vector<ExtrinsicParams> extrinsics_vec{};
    std::vector<double> per_view_errors{};
    double rmse = 0.0;
    double time_seconds = 0.0;
    bool successfully = false;
};


}; // namespace ccl
