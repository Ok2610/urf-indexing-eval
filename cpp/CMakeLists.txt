cmake_minimum_required(VERSION 3.15)

project(ANNInteractiveLearning)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (WITH_MULTITHREAD)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20 -O3 -DANNOYLIB_MULTITHREADED_BUILD")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20 -O3")
endif(WITH_MULTITHREAD)

# HDF5
if (WIN32)
    set(HDF5_DIR "C:/Program Files/HDF_Group/HDF5/1.14.0/cmake") # Points to HDF5Config.cmake
endif (WIN32)
find_package(HDF5 COMPONENTS CXX REQUIRED)
link_directories(${HDF5_LIBRARY_DIRS})
include_directories(${HDF5_INCLUDE_DIRS})

# OpenCV
if (WIN32)
    set(OpenCV_DIR "C:/opencv/sources/build") # Points to OpenCVConfig.cmake 
endif (WIN32)
find_package(OpenCV 4.5.5 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBS})

# Python3
if (WIN32)
    set(PYTHON3_ROOT_DIR "C:/Users/ok261/AppData/Local/Programs/Python/Python311")
endif (WIN32)
find_package (Python3 COMPONENTS Interpreter Development REQUIRED)
link_directories(${Python3_LIBRARY_DIRS})
include_directories(${Python3_INCLUDE_DIRS})  

add_subdirectory(src/)
