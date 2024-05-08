cmake_minimum_required(VERSION 2.8.3)
project(loam_livox)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)

find_package(PCL REQUIRED)
find_package(Boost REQUIRED)


include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

message(STATUS "***** PCL version: ${PCL_VERSION} *****")
####

include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ./include/tools/
  )
add_executable(laser_feature_extractor source/laser_feature_extractor.cpp)

target_link_libraries(laser_feature_extractor
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${CERES_LIBRARIES}
  ${Boost_LIBRARIES}
)



