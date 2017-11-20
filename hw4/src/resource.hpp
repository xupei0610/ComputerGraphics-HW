#ifndef PX_CG_TEXTURE_LOADER_HPP
#define PX_CG_TEXTURE_LOADER_HPP

namespace px
{

static const unsigned char BRICK_DIFFUSE_MAP1[] = {
#include "texture/brick_d1.dat"
};
static const unsigned char BRICK_NORMAL_MAP1[] = {
#include "texture/brick_n1.dat"
};
static const unsigned char BRICK_SPECULAR_MAP1[] = {
#include "texture/brick_s1.dat"
};
static const unsigned char BRICK_DISPLACEMENT_MAP1[] = {
#include "texture/brick_h1.dat"
};
//static const unsigned char BRICK_DIFFUSE_MAP2[] = {
//#include "texture/brick_d2.dat"
//};
//static const unsigned char BRICK_NORMAL_MAP2[] = {
//#include "texture/brick_n2.dat"
//};
//static const unsigned char BRICK_SPECULAR_MAP2[] = {
//#include "texture/brick_s2.dat"
//};
//static const unsigned char BRICK_DISPLACEMENT_MAP2[] = {
//#include "texture/brick_h2.dat"
//};
//static const unsigned char BRICK_DIFFUSE_MAP3[] = {
//#include "texture/brick_d3.dat"
//};
//static const unsigned char BRICK_NORMAL_MAP3[] = {
//#include "texture/brick_n3.dat"
//};
//static const unsigned char BRICK_SPECULAR_MAP3[] = {
//#include "texture/brick_s3.dat"
//};
//static const unsigned char BRICK_DISPLACEMENT_MAP3[] = {
//#include "texture/brick_h3.dat"
//};
//static const unsigned char BRICK_DIFFUSE_MAP4[] = {
//#include "texture/brick_d4.dat"
//};
//static const unsigned char BRICK_NORMAL_MAP4[] = {
//#include "texture/brick_n4.dat"
//};
//static const unsigned char BRICK_SPECULAR_MAP4[] = {
//#include "texture/brick_s4.dat"
//};
//static const unsigned char BRICK_DISPLACEMENT_MAP4[] = {
//#include "texture/brick_h4.dat"
//};
//static const unsigned char BRICK_DIFFUSE_MAP5[] = {
//#include "texture/brick_d5.dat"
//};
//static const unsigned char BRICK_NORMAL_MAP5[] = {
//#include "texture/brick_n5.dat"
//};
//static const unsigned char BRICK_SPECULAR_MAP5[] = {
//#include "texture/brick_s5.dat"
//};
//static const unsigned char BRICK_DISPLACEMENT_MAP5[] = {
//#include "texture/brick_h5.dat"
//};
//static const unsigned char ROCK_DIFFUSE_MAP1[] = {
//#include "texture/rock_d1.dat"
//};
//static const unsigned char ROCK_NORMAL_MAP1[] = {
//#include "texture/rock_n1.dat"
//};
//static const unsigned char ROCK_SPECULAR_MAP1[] = {
//#include "texture/rock_s1.dat"
//};
//static const unsigned char ROCK_DISPLACEMENT_MAP1[] = {
//#include "texture/rock_h1.dat"
//};
//static const unsigned char CONCRETE_DIFFUSE_MAP1[] = {
//#include "texture/concrete_d1.dat"
//};
//static const unsigned char CONCRETE_NORMAL_MAP1[] = {
//#include "texture/concrete_n1.dat"
//};
//static const unsigned char CONCRETE_SPECULAR_MAP1[] = {
//#include "texture/concrete_s1.dat"
//};
//static const unsigned char CONCRETE_DISPLACEMENT_MAP1[1024*1024] = {
//0
//};
//static const unsigned char GRAVEL_DIFFUSE_MAP1[] = {
//#include "texture/gravel_d1.dat"
//};
//static const unsigned char GRAVEL_NORMAL_MAP1[] = {
//#include "texture/gravel_n1.dat"
//};
//static const unsigned char GRAVEL_SPECULAR_MAP1[] = {
//#include "texture/gravel_s1.dat"
//};
//static const unsigned char GRAVEL_DISPLACEMENT_MAP1[] = {
//#include "texture/gravel_h1.dat"
//};

static const unsigned char *WALL_TEXTURES[] = {
        BRICK_DIFFUSE_MAP1, BRICK_NORMAL_MAP1, BRICK_SPECULAR_MAP1, BRICK_DISPLACEMENT_MAP1,
//        BRICK_DIFFUSE_MAP2, BRICK_NORMAL_MAP2, BRICK_SPECULAR_MAP2, BRICK_DISPLACEMENT_MAP2,
//        BRICK_DIFFUSE_MAP3, BRICK_NORMAL_MAP3, BRICK_SPECULAR_MAP3, BRICK_DISPLACEMENT_MAP3,
//        BRICK_DIFFUSE_MAP4, BRICK_NORMAL_MAP4, BRICK_SPECULAR_MAP4, BRICK_DISPLACEMENT_MAP4,
//        BRICK_DIFFUSE_MAP5, BRICK_NORMAL_MAP5, BRICK_SPECULAR_MAP5, BRICK_DISPLACEMENT_MAP5,
//        CONCRETE_DIFFUSE_MAP1, CONCRETE_NORMAL_MAP1, CONCRETE_SPECULAR_MAP1, CONCRETE_DISPLACEMENT_MAP1
};
static const int WALL_TEXTURE_DIM[] = {
        // w  h
        1024, 1024,
//        1024, 1024,
//        1600, 1600,
//        1024, 1024,
//        1024, 1024,
//        1024, 1024
};

static const unsigned char *FLOOR_TEXTURES[] = {
        BRICK_DIFFUSE_MAP1, BRICK_NORMAL_MAP1, BRICK_SPECULAR_MAP1, BRICK_DISPLACEMENT_MAP1,
//        BRICK_DIFFUSE_MAP2, BRICK_NORMAL_MAP2, BRICK_SPECULAR_MAP2, BRICK_DISPLACEMENT_MAP2,
//        ROCK_DIFFUSE_MAP1,  ROCK_NORMAL_MAP1,  ROCK_SPECULAR_MAP1,  ROCK_DISPLACEMENT_MAP1,
//        GRAVEL_DIFFUSE_MAP1,GRAVEL_NORMAL_MAP1,GRAVEL_SPECULAR_MAP1,GRAVEL_DISPLACEMENT_MAP1,
};
static const int FLOOR_TEXTURE_DIM[] = {
        // w  h
        1024, 1024,
//        1024, 1024,
//        1024, 666,
//        1024, 1024
};

static const int N_WALL_TEXTURES = 1;
static const int N_FLOOR_TEXTURES = 1;
}

#endif
