cmake_minimum_required(VERSION 3.2)

project(hw3)

option(USE_OPENMP "Use OpenMP or not." ON)
option(USE_GUI "Use GUI or not" ON)
option(USE_CUDA "Use CUDA or not" ON)
set(USE_CUDA OFF)
option(USE_JITTER "Use Jitter Sampling" OFF)
option(USE_ADAPTIVE "Use Adaptive Sampling" OFF)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_VERBOSE_MAKEFILE TRUE)
set(CMAKE_CXX_FLAGS "-Wall -O3")
set(CMAKE_C_FLAGS "-Wall -O3")
set(EXE_NAME ${PROJECT_NAME})

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
endif ()

if (USE_CUDA)
    find_package(CUDA 5.0.0 REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    add_definitions("-DUSE_CUDA=${USE_CUDA}")
endif()

if (USE_JITTER)
    add_definitions("-DJITTER_SAMPLING=${USE_JITTER}")
endif ()
if (USE_ADAPTIVE)
    add_definitions("-DADAPTIVE_SAMPLING=${USE_ADAPTIVE}")
endif ()

# FIXME for IDE
file(GLOB HEADER_FILES
     ${PROJECT_SOURCE_DIR}/src/object/light/light.hpp
     ${PROJECT_SOURCE_DIR}/src/object/material/*.hpp
     ${PROJECT_SOURCE_DIR}/src/object/geometry/*.hpp
     ${PROJECT_SOURCE_DIR}/src/object/structure/*.hpp
     ${PROJECT_SOURCE_DIR}/src/object/*.hpp
     ${PROJECT_SOURCE_DIR}/src/*.hpp
     )

file(GLOB EXE_COMMON_SRC_FILES
     ## TODO Put objects into dependent files
     ${PROJECT_SOURCE_DIR}/src/object/base_object.cu
     ${PROJECT_SOURCE_DIR}/src/object/light/*.cu
     ${PROJECT_SOURCE_DIR}/src/object/material/*.cu
     ${PROJECT_SOURCE_DIR}/src/object/geometry/*.cu
     ${PROJECT_SOURCE_DIR}/src/object/structure/*.cu
     )

file(GLOB EXE_CUDA_SRC_FILES
     ${PROJECT_SOURCE_DIR}/src/trace.cu
     ${PROJECT_SOURCE_DIR}/src/scene.cu
     )

file(GLOB EXE_CPU_SRC_FILES
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
        -arch=compute_61
        -code=sm_61
        -O3)
    cuda_add_executable(${EXE_NAME}
                        ${HEADER_FILES}
                        ${EXE_COMMON_SRC_FILES}
                        ${EXE_CPU_SRC_FILES}
                        ${EXE_CUDA_SRC_FILES}
                        ${EXE_GUI_FILES}
                        ${PROJECT_SOURCE_DIR}/src/main.cpp)
else()
    set_source_files_properties(${EXE_COMMON_SRC_FILES} PROPERTIES LANGUAGE CXX)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
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
