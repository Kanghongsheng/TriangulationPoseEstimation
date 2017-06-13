cmake_minimum_required(VERSION 2.8)
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )
project(pose_estimation)
find_package(OpenCV 3.2 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(pose_estimation poseEstimation.cpp)
target_link_libraries(pose_estimation ${OpenCV_LIBS})


