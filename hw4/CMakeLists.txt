cmake_minimum_required(VERSION 3.2)
project(hw4)
set(EXE_NAME hw4)

# Setup Comipler Flags
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (MSVC)
    set(CMAKE_CXX_FLAGS "/W4 /O2")
else()
    set(CMAKE_CXX_FLAGS "-Wall -O3")
endif()

set(CMAKE_VERBOSE_MAKEFILE TRUE)
set(CMAKE_MACOSX_RPATH 0)

# Setup Dependencies
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(Freetype REQUIRED)
include_directories(${FREETYPE_INCLUDE_DIRS})
find_package(GLM REQUIRED)
find_package(glfw3 3.2 REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)
if (APPLE)
    include_directories(/usr/local/include)
    find_library(COCOA_LIBRARY Cocoa REQUIRED)
    find_library(IOKIT_LIBRARY IOKit REQUIRED)
    find_library(COREVID_LIBRARY CoreVideo REQUIRED)
endif()
add_subdirectory(${CMAKE_SOURCE_DIR}/lib/assimp/)
include_directories(${CMAKE_SOURCE_DIR}/lib/assimp/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
set(SOURCE_LIBRARIES
    ${CMAKE_THREAD_LIBS_INIT}
    ${FREETYPE_LIBRARIES}
    glfw
    ${GLEW_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${COCOA_LIBRARY} ${IOKIT_LIBRARY} ${COREVID_LIBRARY}
    assimp)

# Setup Sources
include_directories(${CMAKE_SOURCE_DIR}/asset)
include_directories(${CMAKE_SOURCE_DIR}/src)
set(SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)
file(GLOB SOURCE_FILES
     ${SOURCE_DIR}/soil/*.c
     ${SOURCE_DIR}/timer.cpp
     ${SOURCE_DIR}/item.cpp
     ${SOURCE_DIR}/glfw.cpp
     ${SOURCE_DIR}/mesh.cpp
     ${SOURCE_DIR}/shader/base_shader.cpp
     ${SOURCE_DIR}/shader/text.cpp
     ${SOURCE_DIR}/shader/rectangle.cpp
     ${SOURCE_DIR}/shader/skybox.cpp
     ${SOURCE_DIR}/item/key.cpp
     ${SOURCE_DIR}/item/door.cpp
     ${SOURCE_DIR}/option.cpp
     ${SOURCE_DIR}/maze.cpp
     ${SOURCE_DIR}/camera.cpp
     ${SOURCE_DIR}/character.cpp
     ${SOURCE_DIR}/scene.cpp
     ${SOURCE_DIR}/app.cpp)

add_executable(${EXE_NAME}
               ${SOURCE_FILES}
               ${SOURCE_DIR}/main.cpp)
target_link_libraries(${EXE_NAME}
                      ${SOURCE_LIBRARIES})


