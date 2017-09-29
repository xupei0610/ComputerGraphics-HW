cmake_minimum_required(VERSION 3.2)

project(image)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_VERBOSE_MAKEFILE TRUE)
set(CMAKE_CXX_FLAGS "-Wall -Og")
set(CMAKE_C_FLAGS "-Wall -Og")
set(EXE_NAME ${PROJECT_NAME})

file(GLOB EXE_SRC_FILES
    pixel.cpp
    image.cpp
    )

add_executable(${EXE_NAME}
    ${EXE_SRC_FILES}
    main.cpp)

install(TARGETS ${EXE_NAME} DESTINATION ${PROJECT_SOURCE_DIR}/bin)
