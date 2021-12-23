cmake_minimum_required(VERSION 3.10.2)

project(nsfw)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp -fexceptions")

set(NCNN_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR}/ncnn-20210720-ubuntu-1804)

include_directories(${Vulkan_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS}/opencv4)
include_directories(${NCNN_ROOT_DIR}/include)


set(SRC_LIST
	./main.cpp)
add_executable(${CMAKE_PROJECT_NAME} ${SRC_LIST})

FIND_PACKAGE(OpenMP REQUIRED)
if(OPENMP_FOUND)
	message("OPENMP FOUND")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(OpenCV REQUIRED)

find_package(Vulkan REQUIRED)

set(ncnn_DIR ${NCNN_ROOT_DIR}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

target_link_libraries(${CMAKE_PROJECT_NAME} ${OpenCV_LIBS} ncnn ${Vulkan_LIBRARY})