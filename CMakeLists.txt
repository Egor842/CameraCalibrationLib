cmake_minimum_required(VERSION 3.10)
project(ccl)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Ceres REQUIRED)

# Add executable with all source files
add_executable(ccl 
    test_calib2.cpp 
    src/CameraModel.cpp 
    src/BrownConradyModel.cpp 
    src/BrownConradyCalibrator.cpp
    src/SyntheticCalibrator.cpp 
    src/EdgeDetector.cpp 
    src/ChainTree.cpp 
    src/EdgeDetectorParamsFree.cpp 
    src/ChessboardDetector.cpp 
    src/NFA.cpp 
    src/LineDetector.cpp
)

# Link libraries
target_link_libraries(ccl 
    ${OpenCV_LIBS} 
    ${YAML_CPP_LIBRARIES}
    ${CERES_LIBRARIES}
)

# Include directories for Ceres
target_include_directories(ccl PRIVATE
    ${CERES_INCLUDE_DIRS}
)
