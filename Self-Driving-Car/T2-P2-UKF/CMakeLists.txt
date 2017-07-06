cmake_minimum_required(VERSION 3.7)
project(P2_UKF)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        src/ukf.cpp
        src/tools.cpp
        src/main.cpp)
add_executable(P2_UKF ${SOURCE_FILES})