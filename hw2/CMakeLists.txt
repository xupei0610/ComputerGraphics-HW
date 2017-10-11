cmake_minimum_required(VERSION 3.2)

project(hw2)

option(USE_OPENMP "Use OpenMP or not." ON)
option(USE_GUI "Use GUI or not" ON)
option(USE_JITTER "Use Jitter Sampling" OFF)
option(USE_ADAPTIVE "Use Jitter Sampling" OFF)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_VERBOSE_MAKEFILE TRUE)
set(CMAKE_CXX_FLAGS "-Wall -O3")
set(CMAKE_C_FLAGS "-Wall -O3")
set(EXE_NAME ${PROJECT_NAME})

if (USE_GUI)
    add_definitions("-DUSE_GUI=${USE_GUI}")
	include(FindPkgConfig)
	pkg_search_module(SDL2 REQUIRED sdl2)
endif()

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

if (USE_JITTER)
    add_definitions("-DJITTER_SAMPLING=${USE_JITTER}")
endif()
if (USE_ADAPTIVE)
    add_definitions("-DADAPTIVE_SAMPLING=${USE_ADAPTIVE}")
endif()

include_directories(
    ${SDL2_INCLUDE_DIRS}
    )

if (USE_GUI)
  file(GLOB EXE_SRC_FILES
     src/object.cpp
     src/scene.cpp
     src/window.cpp
    )
else()
  file(GLOB EXE_SRC_FILES
     src/object.cpp
     src/scene.cpp
    )
endif()

add_executable(${EXE_NAME}
    ${EXE_SRC_FILES}
    src/main.cpp)

target_link_libraries(${EXE_NAME}
    ${SDL2_LIBRARIES}
    )
