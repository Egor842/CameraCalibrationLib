if(BUILD_SHARED_LIBS)
    add_library(ccl_models SHARED
        CameraModel.cpp
        Distortion.cpp
        ExtrinsicParams.cpp
        IntrinsicParams.cpp
        BrownConradyDistortion.cpp
    )
else()
    add_library(ccl_models STATIC
        CameraModel.cpp
        Distortion.cpp
        ExtrinsicParams.cpp
        IntrinsicParams.cpp
        BrownConradyDistortion.cpp
    )
endif()

target_include_directories(ccl_models
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

if(OpenCV_FOUND)
    target_include_directories(ccl_calibrators SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(ccl_calibrators PUBLIC OpenCV::OpenCV)
endif()
if(yaml-cpp_FOUND)
    target_link_libraries(ccl_calibrators PUBLIC yaml-cpp::yaml-cpp)
endif()
