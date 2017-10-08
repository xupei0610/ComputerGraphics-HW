cmake_minimum_required(VERSION 3.2)

project(image)

option(USE_OPENMP "Use OpenMP or not." ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_VERBOSE_MAKEFILE TRUE)
set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_C_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_C_FLAGS_RELEASE "-O3")
set(EXE_NAME ${PROJECT_NAME})

if (USE_OPENMP)
    find_package(OpenMP)
    if (OPENMP_FOUND)
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(PNET_USE_OPENMP ON)
    else()
        message (STATUS "No OpenMP support found.")
    endif()
endif()

file(GLOB EXE_SRC_FILES
    src/pixel.cpp
    src/image.cpp
    )

add_executable(${EXE_NAME}
    ${EXE_SRC_FILES}
    main.cpp)

target_link_libraries(${EXE_NAME})

install(TARGETS ${EXE_NAME} DESTINATION ${PROJECT_SOURCE_DIR}/bin)
