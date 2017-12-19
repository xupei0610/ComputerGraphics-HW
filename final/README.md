# Final

This is the final project for Fundamentals of Computer Graphics I.

## Usage

    mkdir build
    cd build
    cmake ..
    make
    ./final

To use `-DDISABLE_MULTISAMPLE` with `cmake` command to disable multisampling.

Dependencies: FreeType, GLFW, GLM, GLEW

3rd Part libraries: SOIL, ASSIMP

Tested under Ubuntu 16.04 and Mac OS 10.12. Retia Display supported.

## Introduction

This project is the extendation of HW4. This project focuses more on the implementation of deferred rendering. Two main features are implemented: **deferred rendering** with bumping and displacement mapping; and **shootable light balls** whose movement follows basic physical rules.

## Deferred Rendering

The performance of deferred rendering is amazing. The following shows a scene with hundreds of randomly moving light balls drawn by deferred rendering.

<img src="./doc/lights2.gif" />

<img src="./doc/lights.png" />

The followings show the scence in game. Light balls floating in the air and move slowly and randomly, like fireflies.

<img src="./doc/lights3.png" />

<img src="./doc/lights4.png" />

<img src="./doc/lights5.png" />

<img src="./doc/lights6.png" />

<img src="./doc/lights7.png" />

The floating light balls are allowed to go through walls. The effect when they go through walls looks very good.

<img src="./doc/lights8.gif" />

In the game scene, all objects are rendered by diffuse, normal, specular and displacement/fieldheight mapping. Before introduction of deferred rendering, all these mappings are finished in tangent space for convenience. With deferred rendering, all these mappings have to be finished in the world or view coordinates; otherwise, we have to pass the tangent-bitangent-normal matrix through buffers for the sake of transfering the light and view position into the tangent space. Our choice in the project is to convert all things into the world frame such that the amount of computation reduces largely when rendering lights.

After the introduction of deferred rendering, the process of rendering scence can be splitted into three parts:

1. render objects normally with the head light and store material(diffuse, specular), position, and normal attributes per pixel into framebuffers;
2. render lights for each pixel based on the attributes passed by framebuffers;
3. render objects not influenced by light, e.g. light source objects and the sky box.

## Shootable Light Balls

Besides that they are projected by the player, the shootable light balls has another important feature, that are different to the floating light balls: their movement follows the basic physical rules. They could fall down, could slide, could bounce, and could lose velocity during movement.

<img src="./doc/shoot.gif" />

<img src="./doc/shoot2.gif" />

<img src="./doc/shoot3.gif" />

<img src="./doc/shoot4.png" />

## Deferred Rendering with Post Processing

In the pause screen, the current scene is convolved with a Gaussian kernel and used as the background. 

<img src="./doc/post.png" />

## Additional Work

A simple animation effect, when the player opens a door.

<img src="./doc/animation.gif" />


