cmake_minimum_required(VERSION 3.2)

project(hw0.5)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_VERBOSE_MAKEFILE TRUE)
set(CMAKE_CXX_FLAGS "-Wall -O3")
set(CMAKE_C_FLAGS "-Wall -O3")

find_package(OpenGL REQUIRED)
include(FindPkgConfig)
pkg_search_module(SDL2 REQUIRED sdl2)

include_directories(${SDL2_INCLUDE_DIRS})
include_directories(${OPENGL_INCLUDE_DIRS})

set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)
file(GLOB SRC_FILES
    ${SRC_DIR}/glad/glad.c
)

add_executable(BlankScreenSDL
    ${SRC_FILES}
    ${SRC_DIR}/BlankScreenSDL.cpp
)
add_executable(Cube3D
    ${SRC_FILES}
    ${SRC_DIR}/Cube3D.cpp
)
add_executable(CubeLit
    ${SRC_FILES}
    ${SRC_DIR}/CubeLit.cpp
)
add_executable(CubeLit1VBO
    ${SRC_FILES}
    ${SRC_DIR}/CubeLit1VBO.cpp
)
add_executable(ModelLoad
    ${SRC_FILES}
    ${SRC_DIR}/ModelLoad.cpp
)
add_executable(TriangleColored
    ${SRC_FILES}
    ${SRC_DIR}/TriangleColored.cpp
)

set(SRC_LIBS
    dl
    ${OPENGL_LIBRARIES}
    ${SDL2_LIBRARIES}
)
target_link_libraries(BlankScreenSDL  ${SRC_LIBS})
target_link_libraries(Cube3D          ${SRC_LIBS})
target_link_libraries(CubeLit         ${SRC_LIBS})
target_link_libraries(CubeLit1VBO     ${SRC_LIBS})
target_link_libraries(ModelLoad       ${SRC_LIBS})
target_link_libraries(TriangleColored ${SRC_LIBS})

add_executable(ModelLoadNew
               ${SRC_FILES}
               ${SRC_DIR}/ModelLoadNew.cpp
               )
add_executable(TriangleColoredNew
               ${SRC_FILES}
               ${SRC_DIR}/TriangleColoredNew.cpp
               )
target_link_libraries(TriangleColoredNew ${SRC_LIBS})
target_link_libraries(ModelLoadNew       ${SRC_LIBS})