cmake_minimum_required(VERSION 3.15)

set(BINARY ${CMAKE_PROJECT_NAME})

file(GLOB_RECURSE SOURCES LIST_DIRECTORIES true interactivelearning/annoy/*.h interactivelearning/exq/*.h interactivelearning/exq/*.cpp interactivelearning/hnsw/*.h)

set(SOURCES ${SOURCES})

add_library(${BINARY}_lib STATIC ${SOURCES})

add_executable(${BINARY}_run main.cpp) #${SOURCES})

target_link_libraries(${BINARY}_run ${BINARY}_lib)
target_link_libraries(${BINARY}_run ${HDF5_CXX_LIBRARIES})
target_link_libraries(${BINARY}_run ${OpenCV_LIBS})
target_link_libraries(${BINARY}_run ${Python3_LIBRARIES})
