cmake_minimum_required(VERSION 3.2)

project(hw2)

option(USE_OPENMP "Use OpenMP or not." ON)
option(JITTER "Use Jitter Sampling" OFF)
option(ADAPTIVE "Use Jitter Sampling" OFF)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_VERBOSE_MAKEFILE TRUE)
set(CMAKE_CXX_FLAGS "-Wall -O3")
set(CMAKE_C_FLAGS "-Wall -O3")
set(EXE_NAME ${PROJECT_NAME})

include(FindPkgConfig)
pkg_search_module(SDL2 REQUIRED sdl2)

if (USE_OPENMP)
    find_package(OpenMP)
    if (OPENMP_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(PNET_USE_OPENMP ON)
    else()
        message (STATUS "No OpenMP support found.")
    endif()
endif()

if (JITTER)
    add_definitions("-DJITTER_SAMPLING=${JITTER}")
endif()
if (ADAPTIVE)
    add_definitions("-DADAPTIVE_SAMPLING=${ADAPTIVE}")
endif()

include_directories(
    ${SDL2_INCLUDE_DIRS}
    )

file(GLOB EXE_SRC_FILES
     src/object.cpp
     src/scene.cpp
     src/window.cpp
    )

add_executable(${EXE_NAME}
    ${EXE_SRC_FILES}
    src/main.cpp)

target_link_libraries(${EXE_NAME}
    ${SDL2_LIBRARIES}
    )
