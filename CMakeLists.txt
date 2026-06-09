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

add_library(${PROJECT_NAME} INTERFACE)
target_link_libraries(${PROJECT_NAME} INTERFACE
    ccl_models
    ccl_detectors
    ccl_calibrators
    ccl_utility
)
target_include_directories(${PROJECT_NAME} INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

include(GNUInstallDirs)

install(TARGETS
    ccl_models
    ccl_detectors
    ccl_calibrators
    ccl_utility
    ${PROJECT_NAME}
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

include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cclConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/cclConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ccl
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/cclConfig.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ccl
)
