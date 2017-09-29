## HW0 Demo

This code is for HW0 of CSci 5607 Fundamentals of Computer Graphics, UMN. This is a **basic** version of HW0, which only implements the basic, required features through directly adding code at the specified location in the provided demo code.

A complete version, which implements all requried and extra features, can be found at

<https://github.com/xupei0610/ComputerGraphics-HW/tree/master/hw0>

## Usage

    mkdir build
    cd build
    cmake ..
    make
    ./hw0

## Description

This a simple OpenGL demo with SDL2. It draws a red sqaure and implements three kinds of basic interaction:

+ **Translation**: drag mouse inside the square
+ **Scaling**: drag mouse on any edge of the square
+ **Rotation**: drag mouse on any corner of the square

When doing translation, the square moves following the mouse cursor smoothly instead of directly jumping to the cursor position.

Some margions are considered, when checking the relative position of the mouse cursor to the square, so that the user does not really need to put the mouse at the edge or corner of the square.

