cmake_minimum_required(VERSION 3.2)

project(hw3)

option(USE_OPENMP "Use OpenMP or not." ON)
option(USE_GUI "Use GUI or not" ON)
set(USE_GUI OFF)
option(USE_CUDA "Use CUDA or not" ON)
#set(USE_CUDA OFF)
option(USE_JITTER "Use Jitter Sampling" OFF)
option(USE_ADAPTIVE "Use Adaptive Sampling" OFF)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_VERBOSE_MAKEFILE TRUE)
set(CMAKE_CXX_FLAGS "-Wall -O3")
set(CMAKE_C_FLAGS "-Wall -O3")
set(CMAKE_CXX_FLAGS_RELEASE "")
set(CMAKE_CXX_FLAGS_DEBUG "")
set(CMAKE_C_FLAGS_RELEASE "")
set(CMAKE_C_FLAGS_DEBUG "")

if (MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /FS")
	set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} /O2 /FS")
endif()

set(EXE_NAME ${PROJECT_NAME})

add_definitions("-DPREC=float")

find_package(Threads)
include_directories(${PROJECT_SOURCE_DIR}/src)

if (USE_OPENMP)
    find_package(OpenMP)
    if (OPENMP_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(PNET_USE_OPENMP ON)
    else ()
        message(STATUS "No OpenMP support found.")
    endif ()
endif ()

if (USE_GUI)
    add_definitions("-DUSE_GUI=${USE_GUI}")
    include(FindPkgConfig)
    pkg_search_module(SDL2 REQUIRED sdl2)
    include_directories(${SDL2_INCLUDE_DIRS})
endif()

if (USE_CUDA)
    find_package(CUDA 5.0.0 REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    add_definitions("-DUSE_CUDA=${USE_CUDA}")
endif()

if (USE_JITTER)
    add_definitions("-DJITTER_SAMPLING=${USE_JITTER}")
endif()
if (USE_ADAPTIVE)
    add_definitions("-DADAPTIVE_SAMPLING=${USE_ADAPTIVE}")
endif()

file(GLOB EXE_COMMON_SRC_FILES
     ${PROJECT_SOURCE_DIR}/src/object/light/*.cu
     ${PROJECT_SOURCE_DIR}/src/object/material/*.cu
     ${PROJECT_SOURCE_DIR}/src/object/geometry/base_geometry.cpp
     ${PROJECT_SOURCE_DIR}/src/object/geometry/plane.cu
     ${PROJECT_SOURCE_DIR}/src/object/geometry/disk.cu
     ${PROJECT_SOURCE_DIR}/src/object/geometry/ring.cu
     ${PROJECT_SOURCE_DIR}/src/object/geometry/triangle.cu
     ${PROJECT_SOURCE_DIR}/src/object/geometry/normal_triangle.cu
     ${PROJECT_SOURCE_DIR}/src/object/geometry/box.cu
     ${PROJECT_SOURCE_DIR}/src/object/geometry/sphere.cu
     ${PROJECT_SOURCE_DIR}/src/object/geometry/ellipsoid.cu
     ${PROJECT_SOURCE_DIR}/src/object/geometry/cone.cu
     ${PROJECT_SOURCE_DIR}/src/object/geometry/cylinder.cu
     ${PROJECT_SOURCE_DIR}/src/object/geometry/quadric.cu
     ${PROJECT_SOURCE_DIR}/src/object/structure/bound_box.cu
     ${PROJECT_SOURCE_DIR}/src/object/structure/bvh.cu
     )

file(GLOB EXE_CUDA_SRC_FILES
     ${PROJECT_SOURCE_DIR}/src/trace.cu
     ${PROJECT_SOURCE_DIR}/src/scene.cu
     )

file(GLOB EXE_CPU_SRC_FILES
     ${PROJECT_SOURCE_DIR}/src/object/base_object.cpp
     ${PROJECT_SOURCE_DIR}/src/object/light/base_light.cpp
     ${PROJECT_SOURCE_DIR}/src/object/material/base_material.cpp
     ${PROJECT_SOURCE_DIR}/src/trace.cpp
     ${PROJECT_SOURCE_DIR}/src/scene.cpp
     ${PROJECT_SOURCE_DIR}/src/parser.cpp
     )

if (USE_GUI)
    file(GLOB EXE_GUI_FILES
         ${PROJECT_SOURCE_DIR}/src/window.cpp
         )
endif()

if (USE_CUDA)
    set(CUDA_SEPARABLE_COMPILATION ON)
    set(CUDA_NVCC_FLAGS
        -std=c++11
        -use_fast_math
        -restrict
        --ptxas-options=-O3
        -O3
#        --ptxas-options=-v
#        -res-usage
	    -cl-version=2017
    )
    cuda_add_executable(${EXE_NAME}
                        ${HEADER_FILES}
                        ${EXE_COMMON_SRC_FILES}
                        ${EXE_CPU_SRC_FILES}
                        ${EXE_CUDA_SRC_FILES}
                        ${EXE_GUI_FILES}
                        ${PROJECT_SOURCE_DIR}/src/main.cpp)
else()
    set_source_files_properties(${EXE_COMMON_SRC_FILES} PROPERTIES LANGUAGE CXX)
    if (!MSVC)
		add_definitions("-x c++")
    endif()
	add_executable(${EXE_NAME}
                   ${HEADER_FILES}
                   ${EXE_COMMON_SRC_FILES}
                   ${EXE_CPU_SRC_FILES}
                   ${EXE_GUI_FILES}
                   ${PROJECT_SOURCE_DIR}/src/main.cpp)
endif()

target_link_libraries(${EXE_NAME}
                      ${CMAKE_THREAD_LIBS_INIT}
                      ${SDL2_LIBRARIES}
                      ${CUDA_CUDART_LIBRARY}
                      )


set(CMAKE_BUILD_TYPE "Release")
