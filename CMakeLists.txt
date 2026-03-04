cmake_minimum_required(VERSION 3.10)
project(ccl)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)

add_executable(ccl main.cpp src/EdgeDetector.cpp src/ChainTree.cpp src/EdgeDetectorParamsFree.cpp src/ChessboardDetector.cpp src/NFA.cpp src/LineDetector.cpp)

target_link_libraries(ccl ${OpenCV_LIBS})