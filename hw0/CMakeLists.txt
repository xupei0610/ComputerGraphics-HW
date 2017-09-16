cmake_minimum_required(VERSION 3.2)

project(hw0)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_VERBOSE_MAKEFILE TRUE)
set(CMAKE_CXX_FLAGS "-Wall -O3")
set(CMAKE_C_FLAGS "-Wall -O3")
set(EXE_NAME ${PROJECT_NAME})

find_package(OpenGL REQUIRED)
#find_package(GLUT REQUIRED)
INCLUDE(FindPkgConfig)
PKG_SEARCH_MODULE(SDL2 REQUIRED sdl2)
include_directories(
    ${OPENGL_INCLUDE_DIRS}
    ${GLUT_INCLUDE_DIRS}
    ${SDL2_INCLUDE_DIRS}
    )

file(GLOB EXE_SRC_FILES
    glad/glad.c
    shape.cpp
    shader.cpp
    app.cpp
    )

add_executable(${EXE_NAME}
    ${EXE_SRC_FILES}
    main.cpp)

target_link_libraries(${EXE_NAME}
    ${OPENGL_LIBRARIES}
    ${GLUT_LIBRARIES}
    ${SDL2_LIBRARIES}
    ${CMAKE_DL_LIBS}
    )

install(TARGETS ${EXE_NAME} DESTINATION ${PROJECT_SOURCE_DIR}/bin)
