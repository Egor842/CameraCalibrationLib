if(BUILD_SHARED_LIBS)
    add_library(ccl_utility SHARED
        SyntheticDataGenerator.cpp
    )
else()
    add_library(ccl_utility STATIC
        SyntheticDataGenerator.cpp
    )
endif()

target_include_directories(ccl_utility
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

if(OpenCV_FOUND)
    target_include_directories(ccl_utility SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(ccl_utility PUBLIC ${OpenCV_LIBS})
endif()

