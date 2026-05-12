cmake_minimum_required(VERSION 3.16)
project(ccl_cli LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(yaml-cpp REQUIRED)

find_library(CCL_DETECT ccl_detectors REQUIRED)
find_library(CCL_UTILITY ccl_utility REQUIRED)
find_library(CCL_CALIB ccl_calibrators REQUIRED)
find_library(CCL_MODELS ccl_models REQUIRED)

add_executable(ccl_cli cli.cpp)

target_link_libraries(ccl_cli PRIVATE
    ${OpenCV_LIBS}  
    ${CCL_CALIB}
    ${CCL_MODELS}
    ${CCL_DETECT}  
    ${CCL_UTILITY}           # ← добавляем твою либу
    Ceres::ceres         # ← нужна, потому что CCALIB её использует
    yaml-cpp::yaml-cpp   # ← исправляем предупреждение
)
