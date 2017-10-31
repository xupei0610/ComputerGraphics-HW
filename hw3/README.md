This is the scripts for HW3 of CSci5607 Computer Graphics, UMN, Fall 2017.

This implements a ray tracing program.

### Usage

    mkdir build
    cd build
    cmake ..
    make
    ./hw3

CMake Compile Options:

  + `USE_GUI`: enable GUI using SDL2, default: `ON` 
  + `USE_OPENMP`: enable OpenMP, default: `ON`
  + `USE_CUDA`: enanble CUDA, default: `ON`
  + `USE_JITTER=<n>`: enable Jitter supersampling, `n` is the number of samplings per pixel; default: `OFF`. No GPU support for Jitter supersampling due to performance reason.
  + `USE_ADAPATIVE`: enable adaptive supersampling. This function is done, but not so smart.  

This code was tested under Ubuntu 16.04 with CUDA 8 and GTX1080 Ti and MacOS 10.12 with CUDA 9 and GT650M (it is an old laptop, XD).

**Attention**: This code is not tested under Windows!! My Windows laptop broke last week. It is still on the way sending to the Product Service Center located in Texas.

### Features List

#### CUDA support

  This program support GPU computation using CUDA. If the program is compiled with the `USE_CUDA` option (default), the program will render using GPU by default. If CPU computation is preferred, put

    mode cpu

  in the scene file to enable CPU computation.

  Stack instead of recursion is used in the GPU code.There are still two problems when using GPU computation in the program.
  
  1. Not fully support for the bound box structure. So far, the bound box structure is allowed to be put into another bound box such that a deep function call may be involved when using a bound box containing another one and cause the GPU kernel crashing. I have got an idea to flatten the warping of bound boxes. I will revise this problem later.
  2. Object Initialization. The CPU code is written in modern C++ design pattern originally. Lots of virtual functions are used, which cause a huge problem when running the code on GPU. In order to use virtual functions, objects must be initialized on the device (GPU) directly instead of copying it to the device from the host side. To initalize objects in the device by `new` or `malloc` operation will cause a huge performance problem.  I will talk about it in details in the summary section. This is the flaw of my code design. I will modify the code later, by maintaining function pointers, like `vtable`, manually, to revise this problem.

#### Interactive User Interface
  
  Progress bar in GUI mode
  <img src="./doc/progress_bar.gif" />

  Progress report in CMD mode
  <img src="./doc/progress_report.gif" />

  **Note**: In GPU computation mode, the progress is approximated.

  Multi-thread technique is used to ensure the program is responsable during rendering. To stop rendering, use `Ctrl+C` in CMD mode or press `x` button on the window in the GUI mode. 

#### Geometries 
  Plane, Disk, Ring, Triangle, Normal Triangle, Box, Sphere, Elliposid, (Semi-)Cone, Cylinder, General Quadric.  

  <img src="./doc/geometry_demo.bmp" />

  Grammar:

    plane p_x p_y p_z norm_x norm_y norm_z 
    # (p_x, p_y, p_z) is a point on the plane
    # (norm_x, norm_y, norm_z) is the norm (no need to normalization)

    disk c_x c_y c_z norm_x norm_y norm_z radius
    # (c_x, c_y, c_z) is the center of the disk

    ring c_x c_y c_z norm_x norm_y norm_z radius1 radius2

    box x1 x2 y1 y2 z1 z2
    
    ellipsoid c_x c_y c_z radius_x radius_y radius_z

    cone c_x c_y c_z radius_x radius_y ideal_height real_height
    # (c_x, c_y, c_z) is the center of the bottom face of the cone
    # (radius_x, radius_y) is the radius of the bottom face
    # ideal_height is the height of the cone, use a negative value to generate an inverted cone
    # real_height is the real height of the cone
    # when real_height < ideal_height, the cone will have a flat top face

    cylinder c_x c_y c_z height
    # when height > 0, (c_x, c_y, c_z) is the center of the bottom face of the cone
    # when height < 0, (c_x, c_y, c_z) is the center of the bottom face of the cone

    quadric c_x c_y c_z a b c d e f g h i j x1 x2 y1 y2 z1 z2
    # (c_x, c_y, c_z) is the center point 
    # ax^2 + by^2 + cz^2 + dxy + exz + fyz + gx + hy + iz + j = 0
    # (x1, x2) is the range of x axis
    # (y1, y2) is the range of y axis
    # (z1, z2) is the range of z axis 

  <img src="./doc/gear.bmp" />
  Demo for Normal Triangle

#### Materials
  Uniform, Checkerboard, Brick, Texture

  <img src="./doc/material_demo.bmp" />

  Grammar:

    checkerboard_material ambient_r ambient_g ambient_b diffuse_r diffuse_g diffuse_b specular_r specular_g specular_b specular_exp trans_r trans_g trans_b ior scale dark_color_scale
    
    brick_material a_r a_g a_b d_r d_g d_b s_r s_g s_b s_exp t_r t_g t_b ior a_r_edge a_g_edge a_b_edge d_r_edge d_g_edge d_b_edge s_r_edge s_g_edge s_b_edge s_exp_edge t_r_edge t_g_edge t_b_edge ior_edge scale edge_width edge_height
    
    texture a_r a_g a_b d_r d_g d_b s_r s_g s_b s_exp t_r t_g t_b ior texture_file format scale_u scale_v
    # format must be rgb or rgba 
  
#### Transformation
  Rotation, Translation

  <img src="./doc/transformation_demo.bmp" />

  Grammar:

    transform scale_x scale_y scale_z rotate_x rotate_y rotate_z translate_x translate_y translate_z
    ...
    transform end

  `rotate_*` are in degree unit. Translation will be done after rotation.
  
  All objects between `transform` and `transform end` will be transformed. It is allowed to nest `transform` in other ones.  

#### Light
  Directional Light, Point Light, Spot Light, Area Light

  <img src="./doc/directional_light.bmp" /> 
  Demo of the direcitonal light, `scenes/triangle/outdoor.scn` with 3x3 supersampling. No shadow bug like the case shown in the class.

  <img src="./doc/shadow_test.bmp" />
  
  The in-class example scene to test the shadow bug.

  <img src="./doc/spot_light.bmp" />
  A scene using the spot light with 3x3 supersampling.

##### Area Light
  
  Using the method of casting multiple rays to implement the area light and soft shadow effect.

  Grammar:

    area_light pos_x pos_y pos_z light_r light_g light_b radius

  and

    area_light_sampling n
  
  to set the number of samplings for area lights. Default: `32`

  <img src="./doc/soft_shadow.bmp" />

#### Supersampling

  Grammar:

    sampling_radius n

  set the sampling radius

  When `sampling_radius` == 1, then 3x3 samplings for each pixel;
  When `sampling_radius` == 2, then 5x5 samplings for each pixel;
  and so on.

  By default, uniform supersampling is employed. When set `USE_JITTER=<n>` during compiling, Jitter supersampling is employed. For each subsampling regions, randomly sampling `n` times. 

#### Monte-Carlo Integration for indirect diffuse
#### BoundBox

#### Other Interesting Demos
<img src="./arm-top.bmp" />

<img src="./watch.bmp" />

### Performance Comparsion

Test environment:

    Ubuntu 16.04, G++ 5.4.1, CUDA 9.0
    Intel i7-7700K, Nvidia GTX1080 Ti

Test case: `test/gear.txt`

\# of triangles: 6676; Recursion Depth: 2

|         | 1 Thread | 2 Thread | 4 Thread | 6 Thread | 8 Thread | GPU 
|----------------------------------------------------------------------
|Original |
|BoundBox |
|Octree   |  IMPLEMENT later

|3x3 Sampling | 1 Thread | 2 Thread | 4 Thread | 6 Thread | 8 Thread | GPU 
|-------------------------------------------------------------------------
|Original |
|BoundBox |
|Octree   |  IMPLEMENT later


### TODO Lists

  + Revise GPU code. Reduce time to transfer data from the host to device.
  + BVH using OCtree.
  + Revise adpative supersampling.
  + Polygon
  + Environment Light
  + Ambient Occlusion
  + CSG
  + Motion Blur
  + Depth of Fields
  + Zoom Lens
  + Bump Mapping
  + Procedual Wood Material
  + Procedual Marble Material
  + Procedual Bump mapping
  + Heightfields

  It looks like a huge project.



