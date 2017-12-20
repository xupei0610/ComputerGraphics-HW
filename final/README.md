# Final Project

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

This project is an extension of HW4. It use the scene built by HW4 but focuses on the implementation of deferred rendering. Two main features are implemented: **deferred rendering** with bumping and displacement mapping; and **shootable light balls** whose movement follows basic physical rules.

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

The floating light balls are not rigid bodies and allowed to go through walls. In fact, their attributes in the game are not similar to any object in the real. They are like spirits and have no real body: no mass, no collision.

<img src="./doc/lights8.gif" />

In the game scene, all objects are rendered by diffuse, normal, specular and displacement/fieldheight mapping. Before introduction of deferred rendering, all these mappings are done in tangent space for convenience. With deferred rendering, all these mappings have to be done in the world or view coordinates; otherwise, we must pass the tangent-bitangent-normal matrix through buffers for the sake of transfering the light and view position into the tangent space. Our choice in the project is to convert all things into the world frame such that the amount of computation reduces largely when rendering lights.

With the introduction of deferred rendering, the process of rendering scence can be splitted into three parts:

1. render objects normally with the head light and store material(diffuse, specular), position, and normal attributes per pixel into framebuffers;
2. render lights for each pixel based on the attributes passed by framebuffers;
3. render objects not influenced by light, e.g. light source objects and the sky box.

In the game, the light balls and skybox are rendered in step 3 while the walls and floor are rendered in step 1.

An optional process is to postprocess the output of the above processes before we output the scene to the screen. In that case, we put the output of the above three processess into a framebuffer and then postprocess the data stored in the buffer just like to process an image.

A further improvement is to compute the lighting volume or say the effective lighting radius firstly such that we can just skip it when calculating lighting for those objects who are too far from the light source.


## Shootable Light Balls

Besides that they are projected by the player, the shootable light balls have another important feature, that is different to the floating light balls shown above: they are rigid bodies. Their movement follows basic physical rules. They could collide with other rigid bodies, could fall down, could slide, could bounce, and could lose velocity during movement due to fricition or collision with non-rigid body, the air.

<img src="./doc/shoot.gif" />

<img src="./doc/shoot2.gif" />

<img src="./doc/shoot3.gif" />

<img src="./doc/shoot4.png" />

We can see that after collision, the casted light balls would not sticky to or go through the walls. When hitting the wall, they would directly fall down along the wall, or bounce to another direction depending on their mass and  movement trail and velocity before hitting. Similarly, when hitting the floor, the balls would stop movement immediately or after some sliding. No matter what direction at which a light ball is projected originally, it would fall down finally due to gravity and stop movement finally due to resistance caused by fricition or collision with air.

This system is a little complex although it is quite simpler compared to physical engines. All objects have their own mass, velocity, acceleration, motion direction, coefficient of fricition and some other physical attributes. They would interact with other objects during directly or indirectly contact, the objects including the character controlled by the player, although the character is not drawn during rendering. On the other hand, this is a game. It is allowed to appear some objects that do not follow the rules of the real world. Therefore, objects in the game would have some supernatrual attributes, e.g. zero or even negative mass, no real body and the like.

An interesting thing is to do the collsion between small objects (the casted light balls) and the character controlled by the player. The character is described as a cube box when detecting collsion such that a collsion would be detected when the character encounters a small object on the floor and then the motion of the character would be stopped. Image that you are blocked by a pebble on the road. In order to solve this problem, the character is alowed to go through objects whose height is less than 25% of the character's height. 

## Deferred Rendering with Post Processing

In the pause screen, the current scene is convolved with a Gaussian kernel and used as the background. 

<img src="./doc/post.png" />



