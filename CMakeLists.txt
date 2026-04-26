cmake_minimum_required(VERSION 3.20)

set(PROJECT_NAME ccl)
project(${PROJECT_NAME} VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)

option(AUTO_FETCH_OPENCV   "Auto-fetch OpenCV"   OFF)
option(AUTO_FETCH_YAMLCPP  "Auto-fetch yaml-cpp" OFF)
option(AUTO_FETCH_CERES    "Auto-fetch Ceres"    OFF)

function(find_or_fetch pkg_name fetch_flag)
    find_package(${pkg_name} QUIET)
    if(${${pkg_name}_FOUND})
        message(STATUS "${pkg_name} found (version ${${pkg_name}_VERSION})")
    else()
        if(NOT ${fetch_flag})
            message(FATAL_ERROR
                "${pkg_name} not found.\n"
                "  Either install it manually, or enable auto-fetch:\n"
                "    cmake -B build -D${fetch_flag}=ON"
            )
        else()
            message(STATUS "Fetching ${pkg_name} from the internet...")
            include(FetchContent)
            if(pkg_name STREQUAL "yaml-cpp")
                FetchContent_Declare(yamlcpp
                    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp
                    GIT_TAG        yaml-cpp-0.7.0
                )
                FetchContent_MakeAvailable(yamlcpp)
                set(yaml-cpp_FOUND TRUE CACHE INTERNAL "")
            elseif(pkg_name STREQUAL "OpenCV")
                message(WARNING "Auto-fetch for OpenCV is not implemented; please install OpenCV manually.")
            elseif(pkg_name STREQUAL "Ceres")
                message(WARNING "Auto-fetch for Ceres is not implemented; please install Ceres manually.")
            endif()
        endif()
    endif()
endfunction()

find_or_fetch(OpenCV   AUTO_FETCH_OPENCV)
find_or_fetch(yaml-cpp AUTO_FETCH_YAMLCPP)
find_or_fetch(Ceres    AUTO_FETCH_CERES)

add_subdirectory(src/models)
add_subdirectory(src/detectors)
add_subdirectory(src/calibrators)
add_subdirectory(src/utility)

if(BUILD_SHARED_LIBS)
    add_library(${PROJECT_NAME} SHARED)
else()
    add_library(${PROJECT_NAME} STATIC)
endif()

target_sources(${PROJECT_NAME} PRIVATE
    $<TARGET_OBJECTS:ccl_models>
    $<TARGET_OBJECTS:ccl_detectors>
    $<TARGET_OBJECTS:ccl_calibrators>
    $<TARGET_OBJECTS:ccl_utility>
)

if(OpenCV_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS})
endif()
if(yaml-cpp_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC yaml-cpp)
endif()
if(Ceres_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC ${CERES_LIBRARIES})
endif()

target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if(BUILD_SHARED_LIBS)
    include(GenerateExportHeader)
    generate_export_header(${PROJECT_NAME}
        BASE_NAME ccl
        EXPORT_MACRO_NAME CCL_API
        EXPORT_FILE_NAME ${CMAKE_CURRENT_BINARY_DIR}/ccl_export.h
    )
    target_include_directories(${PROJECT_NAME} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    )
endif()

include(GNUInstallDirs)

install(TARGETS ${PROJECT_NAME}
    EXPORT cclTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY
    include/ccl/models
    include/ccl/detectors
    include/ccl/calibrators
    include/ccl/utility
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ccl
)

install(EXPORT cclTargets
    FILE cclTargets.cmake
    NAMESPACE ccl::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ccl
)
