if(BUILD_SHARED_LIBS)
    add_library(ccl_calibrators SHARED
        ZhangCalibrator.cpp
    )
else()
    add_library(ccl_calibrators STATIC
        ZhangCalibrator.cpp
    )
endif()

target_include_directories(ccl_calibrators
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

if(OpenCV_FOUND)
    target_include_directories(ccl_calibrators SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(ccl_calibrators PUBLIC ${OpenCV_LIBS})
endif()
