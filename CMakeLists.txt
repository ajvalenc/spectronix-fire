cmake_minimum_required(VERSION 3.10 FATAL_ERROR) 
project(main)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_library(fire fire.cpp)

add_executable(main main.cpp)

target_link_libraries(main fire ${OpenCV_LIBS})

set_property(TARGET main PROPERTY CXX_STANDARD 17)
