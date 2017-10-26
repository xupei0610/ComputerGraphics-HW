#ifndef PX_CG_GPU_CREATOR_HPP
#define PX_CG_GPU_CREATOR_HPP

#include "util/cuda.hpp"
#include "object/light.hpp"
#include "object/material.hpp"
#include "object/geometry.hpp"
#include "object/structure.hpp"

namespace px { namespace GpuCreator {

__global__
void destroyLight(BaseLight **dev_ptr);
void destroy(BaseLight** const &dev_ptr);

__global__
void destroyGeometry(BaseGeometry **dev_ptr);
void destroy(BaseGeometry** const &dev_ptr);

__global__
void destroyMaterial(BaseMaterial** dev_ptr);
void destroy(BaseMaterial ** const &dev_ptr);

__global__
void createDirectionalLight(BaseLight** dev_ptr, Light light, Direction dir);
void DirectionalLight(BaseLight** const &dev_ptr,
                      Light const &light, Direction const &dir);

__global__
void createPointLight(BaseLight** dev_ptr, Light light, Point pos);
void PointLight(BaseLight** const &dev_ptr,
                Light const &light, Point const &pos);

__global__
void createSpotLight(BaseLight** dev_ptr,
                     Light light,
                     Point pos,
                     Direction direction,
                     PREC half_angle1,
                     PREC half_angle2,
                     PREC falloff);
void SpotLight(BaseLight** const &dev_ptr,
               Light const &light,
               Point const &pos,
               Direction const &direction,
               PREC const &half_angle1,
               PREC const &half_angle2,
               PREC const &falloff);
__global__
void createAreaLight(BaseLight** dev_ptr,
                     Light light,
                     Point center,
                     PREC radius);
void AreaLight(BaseLight** const &dev_ptr,
               Light const &light,
               Point const &center,
               PREC const &radius);

__global__
void createBrickMaterial(BaseMaterial** dev_ptr,
                         Light ambient,
                         Light diffuse,
                         Light specular,
                         int specular_exponent,
                         Light transmissive,
                         PREC refractive_index,
                         Light ambient_edge,
                         Light diffuse_edge,
                         Light specular_edge,
                         int specular_exponent_edge,
                         Light transmissive_edge,
                         PREC refractive_index_edge,
                         PREC scale,
                         PREC edge_width,
                         PREC edge_height,
                         BumpMapping * bump_mapping_dev_ptr);
void BrickMaterial(BaseMaterial** const &dev_ptr,
                   Light const &ambient,
                   Light const &diffuse,
                   Light const &specular,
                   int const &specular_exponent,
                   Light const &transmissive,
                   PREC const &refractive_index,
                   Light const &ambient_edge,
                   Light const &diffuse_edge,
                   Light const &specular_edge,
                   int const &specular_exponent_edge,
                   Light const &transmissive_edge,
                   PREC const &refractive_index_edge,
                   PREC const &scale,
                   PREC const &edge_width,
                   PREC const &edge_height,
                   BumpMapping * const &bump_mapping_dev_ptr);

__global__
void createCheckerBoard(BaseMaterial** dev_ptr,
                        Light ambient,
                        Light diffuse,
                        Light specular,
                        int specular_exponent,
                        Light transmissive,
                        PREC refractive_index,
                        PREC dim_scale,
                        PREC color_scale,
                        BumpMapping * bump_mapping_dev_ptr);
void CheckerboardMaterial(BaseMaterial** const &dev_ptr,
                          Light const &ambient,
                          Light const &diffuse,
                          Light const &specular,
                          int const &specular_exponent,
                          Light const &transmissive,
                          PREC const &refractive_index,
                          PREC const &dim_scale,
                          PREC const &color_scale,
                          BumpMapping * const &bump_mapping_dev_ptr);
__global__
void createUniformMaterial(BaseMaterial **dev_ptr,
                           Light ambient,
                           Light diffuse,
                           Light specular,
                           int specular_exponent,
                           Light transmissive,
                           PREC refractive_index,
                           BumpMapping *bump_mapping_dev_ptr);
void UniformMateiral(BaseMaterial **const &dev_ptr,
                     Light const &ambient,
                     Light const &diffuse,
                     Light const &specular,
                     int const &specular_exponent,
                     Light const &transmissive,
                     PREC const &refractive_index,
                     BumpMapping * bump_mapping_dev_ptr);
__global__
void createTextureMaterial(BaseMaterial **dev_ptr,
                           Light ambient,
                           Light diffuse,
                           Light specular,
                           int specular_exponent,
                           Light transmissive,
                           PREC refractive_index,
                           Texture *texture_dev_ptr,
                           BumpMapping *bump_mapping_dev_ptr);
void TextureMaterial(BaseMaterial **const &dev_ptr,
                     Light const &ambient,
                     Light const &diffuse,
                     Light const &specular,
                     int const &specular_exponent,
                     Light const &transmissive,
                     PREC const &refractive_index,
                     Texture * const &texture_dev_ptr,
                     BumpMapping * const &bump_mapping_dev_ptr);
__global__
void createBox(BaseGeometry **dev_ptr,
               PREC x1, PREC x2, PREC y1, PREC y2, PREC z1, PREC z2,
               BaseMaterial **material_dev_ptr,
               Transformation *transformation_ptr);
void Box(BaseGeometry **const &dev_ptr,
         PREC const &x1, PREC const &x2,
         PREC const &y1, PREC const &y2,
         PREC const &z1, PREC const &z2,
         BaseMaterial **const &material_dev_ptr,
         Transformation *const &transformation_dev_ptr);
__global__
void createCone(BaseGeometry ** dev_ptr,
                Point center_of_bottom_face,
                PREC radius_x,
                PREC radius_y,
                PREC ideal_height,
                PREC real_height,
                BaseMaterial **material_dev_ptr,
                Transformation *trans_dev_ptr);
void Cone(BaseGeometry ** const &dev_ptr,
          Point const &center_of_bottom_face,
          PREC const &radius_x,
          PREC const &radius_y,
          PREC const &ideal_height,
          PREC const &real_height,
          BaseMaterial **const &material_dev_ptr,
          Transformation *const &trans_dev_ptr);

__global__
void createCylinder(BaseGeometry **dev_ptr,
                    Point center_of_bottom_face,
                    PREC radius_x,
                    PREC radius_y,
                    PREC height,
                    BaseMaterial **material_dev_ptr,
                    Transformation *trans_dev_ptr);
void Cylinder(BaseGeometry ** const &dev_ptr,
              Point const &center_of_bottom_face,
              PREC const &radius_x,
              PREC const &radius_y,
              PREC const &height,
              BaseMaterial **const &material_dev_ptr,
              Transformation *const &trans_dev_ptr);

__global__
void createDisk(BaseGeometry ** dev_ptr,
                Point position,
                Direction norm_vec,
                PREC radius,
                BaseMaterial **material_dev_ptr,
                Transformation *trans_dev_ptr);
void Disk(BaseGeometry ** const &dev_ptr,
          Point const &position,
          Direction const &norm_vec,
          PREC const &radius,
          BaseMaterial **const &material_dev_ptr,
          Transformation *const &trans_dev_ptr);
__global__
void createEllipsoid(BaseGeometry ** dev_ptr,
                     Point center,
                     PREC radius_x,
                     PREC radius_y,
                     PREC radius_z,
                     BaseMaterial **material_dev_ptr,
                     Transformation *trans_dev_ptr);
void Ellipsoid(BaseGeometry ** const &dev_ptr,
               Point const &center,
               PREC const &radius_x,
               PREC const &radius_y,
               PREC const &radius_z,
               BaseMaterial **const &material_dev_ptr,
               Transformation *const &trans_dev_ptr);

__global__
void createNormalTriangle(BaseGeometry ** dev_ptr,
                          Point vertex1, Direction normal1,
                          Point vertex2, Direction normal2,
                          Point vertex3, Direction normal3,
                          BaseMaterial **material_dev_ptr,
                          Transformation *trans_dev_ptr);
void NormalTriangle(BaseGeometry ** const &dev_ptr,
                    Point const &vertex1, Direction const &normal1,
                    Point const &vertex2, Direction const &normal2,
                    Point const &vertex3, Direction const &normal3,
                    BaseMaterial **const &material_dev_ptr,
                    Transformation *const &trans_dev_ptr);

__global__
void createPlane(BaseGeometry ** dev_ptr,
                 Point pos,
                 Direction norm_vec,
                 BaseMaterial **material_dev_ptr,
                 Transformation *trans_dev_ptr);
void Plane(BaseGeometry ** const &dev_ptr,
           Point const &pos,
           Direction const &norm_vec,
           BaseMaterial **const &material_dev_ptr,
           Transformation *const &trans_dev_ptr);

__global__
void createQuadric(BaseGeometry ** dev_ptr,
                   Point center,
                   PREC a,
                   PREC b,
                   PREC c,
                   PREC d,
                   PREC e,
                   PREC f,
                   PREC g,
                   PREC h,
                   PREC i,
                   PREC j,
                   PREC x0, PREC x1,
                   PREC y0, PREC y1,
                   PREC z0, PREC z1,
                   BaseMaterial **material_dev_ptr,
                   Transformation *trans_dev_ptr);
void Quadric(BaseGeometry ** const &dev_ptr,
             Point const &center,
             PREC const &a,
             PREC const &b,
             PREC const &c,
             PREC const &d,
             PREC const &e,
             PREC const &f,
             PREC const &g,
             PREC const &h,
             PREC const &i,
             PREC const &j,
             PREC const &x0, PREC const &x1,
             PREC const &y0, PREC const &y1,
             PREC const &z0, PREC const &z1,
             BaseMaterial **const &material_dev_ptr,
             Transformation *const &trans_dev_ptr);

__global__
void createRing(BaseGeometry ** dev_ptr,
                Point pos,
                Direction norm_vec,
                PREC radius1,
                PREC radius2,
                BaseMaterial **material_dev_ptr,
                Transformation *trans_dev_ptr);

void Ring(BaseGeometry ** const &dev_ptr,
          Point const &pos,
          Direction const &norm_vec,
          PREC const &radius1,
          PREC const &radius2,
          BaseMaterial **const &material_dev_ptr,
          Transformation *const &trans_dev_ptr);

__global__
void createSphere(BaseGeometry ** dev_ptr,
                  Point pos,
                  PREC radius,
                  BaseMaterial **material_dev_ptr,
                  Transformation *trans_dev_ptr);

void Sphere(BaseGeometry ** const &dev_ptr,
            Point const &pos,
            PREC const &radius,
            BaseMaterial **const &material_dev_ptr,
            Transformation *const &trans_dev_ptr);
__global__
void createTriangle(BaseGeometry ** dev_ptr,
                    Point a,
                    Point b,
                    Point c,
                    BaseMaterial **material_dev_ptr,
                    Transformation *trans_dev_ptr);

void Triangle(BaseGeometry ** const &dev_ptr,
              Point const &a,
              Point const &b,
              Point const &c,
              BaseMaterial **const &material_dev_ptr,
              Transformation *const &trans_dev_ptr);
__global__
void createBoundBox(BaseGeometry **dev_ptr,
                    BaseGeometry ***objs,
                    int n,
                    Transformation *trans_dev_ptr);

void BoundBox(BaseGeometry** const &dev_ptr,
              BaseGeometry *** const &objs,
              int const &n,
              Transformation *const &trans_dev_ptr);
__global__
void createBVH(BaseGeometry **dev_ptr,
               BaseGeometry ***objs,
               int n);
void BVH(BaseGeometry **const &dev_ptr,
         BaseGeometry ***const &objs,
         int const &n);

}}

#endif // PX_CG_GPU_CREATOR_HPP
