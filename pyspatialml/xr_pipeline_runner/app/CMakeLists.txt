cmake_minimum_required(VERSION 3.22)
project(pyspatialml_xr_runner)

set(SPATIALML_XR_UTILS_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../../../external/SpatialML-XR-Utils"
    CACHE PATH "Path to SpatialML-XR-Utils checkout")
include("${SPATIALML_XR_UTILS_DIR}/cmake/SpatialMLXRUtils.cmake")

spatialml_xr_project(
    TARGET ${PROJECT_NAME}
    SOURCES ${CMAKE_CURRENT_LIST_DIR}/src/main/cpp/xr_pipeline_runner.cpp
)
