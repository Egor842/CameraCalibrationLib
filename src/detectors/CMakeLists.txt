if(BUILD_SHARED_LIBS)
    add_library(ccl_detectors SHARED
        ChessboardDetector.cpp
    )
else()
    add_library(ccl_detectors STATIC
        ChessboardDetector.cpp
    )
endif()

target_include_directories(ccl_detectors
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

if(OpenCV_FOUND)
    target_include_directories(ccl_detectors SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(ccl_detectors PUBLIC ${OpenCV_LIBS})
endif()

