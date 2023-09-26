cmake_minimum_required(VERSION 3.12 FATAL_ERROR) 
project(main)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)

add_library(fire src/fire.cpp)

add_executable(demo_fire demo_fire.cpp)

target_link_libraries(fire ${OpenCV_LIBS})
target_link_libraries(demo_fire fire)

set_property(TARGET demo_fire PROPERTY CXX_STANDARD 14)
