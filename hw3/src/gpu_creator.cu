#include "gpu_creator.hpp"

using namespace px;

__global__
void GpuCreator::destroyLight(BaseLight **dev_ptr)
{
    delete *dev_ptr;
}
void GpuCreator::destroy(BaseLight** const &dev_ptr)
{
    destroyLight<<<1, 1>>>(dev_ptr);
}
__global__
void GpuCreator::destroyGeometry(BaseGeometry **dev_ptr)
{
    delete *dev_ptr;
}
void GpuCreator::destroy(BaseGeometry** const &dev_ptr)
{
    destroyGeometry<<<1, 1>>>(dev_ptr);
}
__global__
void GpuCreator::destroyMaterial(BaseMaterial** dev_ptr)
{
    delete *dev_ptr;
}
void GpuCreator::destroy(BaseMaterial ** const &dev_ptr)
{
    destroyMaterial<<<1, 1>>>(dev_ptr);
}

__global__
void GpuCreator::createDirectionalLight(BaseLight** dev_ptr, Light light, Direction dir)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        (*dev_ptr) = new px::DirectionalLight(light, dir);
    }
}
void GpuCreator::DirectionalLight(BaseLight** const &dev_ptr,
                      Light const &light, Direction const &dir)
{
    createDirectionalLight<<<1, 1>>>(dev_ptr, light, dir);
}

__global__
void GpuCreator::createPointLight(BaseLight** dev_ptr, Light light, Point pos)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        (*dev_ptr) = new px::PointLight(light, pos);
    }
}

void GpuCreator::PointLight(BaseLight** const &dev_ptr,
                Light const &light, Point const &pos)
{
    createPointLight<<<1, 1>>>(dev_ptr, light, pos);
}

__global__
void GpuCreator::createSpotLight(BaseLight** dev_ptr,
                     Light light,
                     Point pos,
                     Direction direction,
                     PREC half_angle1,
                     PREC half_angle2,
                     PREC falloff)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        (*dev_ptr) = new px::SpotLight(light, pos, direction,
                                   half_angle1, half_angle2,
                                   falloff);
    }
}

void GpuCreator::SpotLight(BaseLight** const &dev_ptr,
               Light const &light,
               Point const &pos,
               Direction const &direction,
               PREC const &half_angle1,
               PREC const &half_angle2,
               PREC const &falloff)
{
    createSpotLight<<<1, 1>>>(dev_ptr,
            light, pos, direction,
            half_angle1, half_angle2,
            falloff);
}

__global__
void GpuCreator::createAreaLight(BaseLight** dev_ptr,
                     Light light,
                     Point center,
                     PREC radius)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        (*dev_ptr) =
                new px::AreaLight(light, center, radius);
    }
}

void GpuCreator::AreaLight(BaseLight** const &dev_ptr,
               Light const &light,
               Point const &center,
               PREC const &radius)
{
    createAreaLight<<<1, 1>>>(dev_ptr, light, center, radius);
}

__global__
void GpuCreator::createBrickMaterial(BaseMaterial** dev_ptr,
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
                         BumpMapping * bump_mapping_dev_ptr)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        (*dev_ptr) = new BaseBrickMaterial(ambient,
                                           diffuse,
                                           specular,
                                           specular_exponent,
                                           transmissive,
                                           refractive_index,
                                           ambient_edge,
                                           diffuse_edge,
                                           specular_edge,
                                           specular_exponent_edge,
                                           transmissive_edge,
                                           refractive_index_edge,
                                           scale,
                                           edge_width,
                                           edge_height,
                                           bump_mapping_dev_ptr);
    }
}

void GpuCreator::BrickMaterial(BaseMaterial** const &dev_ptr,
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
                   BumpMapping * const &bump_mapping_dev_ptr)
{
    createBrickMaterial<<<1, 1>>>(dev_ptr,
            ambient, diffuse,
            specular, specular_exponent,
            transmissive, refractive_index,
            ambient_edge, diffuse_edge,
            specular_edge, specular_exponent_edge,
            transmissive_edge, refractive_index_edge,
            scale, edge_width, edge_height,
            bump_mapping_dev_ptr);
}

__global__
void GpuCreator::createCheckerBoard(BaseMaterial** dev_ptr,
                        Light ambient,
                        Light diffuse,
                        Light specular,
                        int specular_exponent,
                        Light transmissive,
                        PREC refractive_index,
                        PREC dim_scale,
                        PREC color_scale,
                        BumpMapping * bump_mapping_dev_ptr)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        (*dev_ptr) = new BaseCheckerboardMaterial(ambient,
                                                  diffuse,
                                                  specular,
                                                  specular_exponent,
                                                  transmissive,
                                                  refractive_index,
                                                  dim_scale,
                                                  color_scale,
                                                  bump_mapping_dev_ptr);
    }
}

void GpuCreator::CheckerboardMaterial(BaseMaterial** const &dev_ptr,
                          Light const &ambient,
                          Light const &diffuse,
                          Light const &specular,
                          int const &specular_exponent,
                          Light const &transmissive,
                          PREC const &refractive_index,
                          PREC const &dim_scale,
                          PREC const &color_scale,
                          BumpMapping * const &bump_mapping_dev_ptr)
{
    createCheckerBoard<<<1, 1>>>(dev_ptr,
            ambient,
            diffuse,
            specular,
            specular_exponent,
            transmissive,
            refractive_index,
            dim_scale,
            color_scale,
            bump_mapping_dev_ptr);
}

__global__
void GpuCreator::createUniformMaterial(BaseMaterial **dev_ptr,
                           Light ambient,
                           Light diffuse,
                           Light specular,
                           int specular_exponent,
                           Light transmissive,
                           PREC refractive_index,
                           BumpMapping *bump_mapping_dev_ptr)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        (*dev_ptr) = new BaseUniformMaterial(ambient,
                                             diffuse,
                                             specular,
                                             specular_exponent,
                                             transmissive,
                                             refractive_index,
                                             bump_mapping_dev_ptr);
    }
}

void GpuCreator::UniformMateiral(BaseMaterial **const &dev_ptr,
                     Light const &ambient,
                     Light const &diffuse,
                     Light const &specular,
                     int const &specular_exponent,
                     Light const &transmissive,
                     PREC const &refractive_index,
                     BumpMapping * bump_mapping_dev_ptr)
{
    createUniformMaterial<<<1, 1>>>(dev_ptr,
            ambient,
            diffuse,
            specular,
            specular_exponent,
            transmissive,
            refractive_index,
            bump_mapping_dev_ptr);
}

__global__
void GpuCreator::createTextureMaterial(BaseMaterial **dev_ptr,
                           Light ambient,
                           Light diffuse,
                           Light specular,
                           int specular_exponent,
                           Light transmissive,
                           PREC refractive_index,
                           Texture *texture_dev_ptr,
                           BumpMapping *bump_mapping_dev_ptr)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        (*dev_ptr) = new BaseTextureMaterial(ambient,
                                             diffuse,
                                             specular,
                                             specular_exponent,
                                             transmissive,
                                             refractive_index,
                                             texture_dev_ptr,
                                             bump_mapping_dev_ptr);
    }
}

void GpuCreator::TextureMaterial(BaseMaterial **const &dev_ptr,
                     Light const &ambient,
                     Light const &diffuse,
                     Light const &specular,
                     int const &specular_exponent,
                     Light const &transmissive,
                     PREC const &refractive_index,
                     Texture * const &texture_dev_ptr,
                     BumpMapping * const &bump_mapping_dev_ptr)
{
    createTextureMaterial<<<1, 1>>>(dev_ptr,
            ambient,
            diffuse,
            specular,
            specular_exponent,
            transmissive,
            refractive_index,
            texture_dev_ptr,
            bump_mapping_dev_ptr);
}

__global__
void GpuCreator::createBox(BaseGeometry **dev_ptr,
               PREC x1, PREC x2, PREC y1, PREC y2, PREC z1, PREC z2,
               BaseMaterial **material_dev_ptr,
               Transformation *transformation_ptr)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        (*dev_ptr) = new BaseBox(x1, x2, y1, y2, z1, z2,
                                 material_dev_ptr == nullptr ? nullptr : *material_dev_ptr,
                                 transformation_ptr);
    }
}

void GpuCreator::Box(BaseGeometry **const &dev_ptr,
         PREC const &x1, PREC const &x2,
         PREC const &y1, PREC const &y2,
         PREC const &z1, PREC const &z2,
         BaseMaterial **const &material_dev_ptr,
         Transformation *const &transformation_dev_ptr)
{
    createBox<<<1, 1>>>(dev_ptr,
            x1, x2, y1, y2, z1, z2,
            material_dev_ptr, transformation_dev_ptr);
}

__global__
void GpuCreator::createCone(BaseGeometry ** dev_ptr,
                Point center_of_bottom_face,
                PREC radius_x,
                PREC radius_y,
                PREC ideal_height,
                PREC real_height,
                BaseMaterial **material_dev_ptr,
                Transformation *trans_dev_ptr)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        (*dev_ptr) = new BaseCone(center_of_bottom_face,
                                  radius_x, radius_y,
                                  ideal_height, real_height,
                                  material_dev_ptr == nullptr ? nullptr : *material_dev_ptr,
                                  trans_dev_ptr);
    }
}

void GpuCreator::Cone(BaseGeometry ** const &dev_ptr,
          Point const &center_of_bottom_face,
          PREC const &radius_x,
          PREC const &radius_y,
          PREC const &ideal_height,
          PREC const &real_height,
          BaseMaterial **const &material_dev_ptr,
          Transformation *const &trans_dev_ptr)
{
    createCone<<<1, 1>>>(dev_ptr,
            center_of_bottom_face,
            radius_x, radius_y,
            ideal_height, real_height,
            material_dev_ptr,
            trans_dev_ptr);
}

__global__
void GpuCreator::createCylinder(BaseGeometry **dev_ptr,
                    Point center_of_bottom_face,
                    PREC radius_x,
                    PREC radius_y,
                    PREC height,
                    BaseMaterial **material_dev_ptr,
                    Transformation *trans_dev_ptr)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        (*dev_ptr) = new BaseCylinder(center_of_bottom_face,
                                      radius_x, radius_y,
                                      height,
                                      material_dev_ptr == nullptr ? nullptr : *material_dev_ptr,
                                      trans_dev_ptr);
    }
}

void GpuCreator::Cylinder(BaseGeometry ** const &dev_ptr,
              Point const &center_of_bottom_face,
              PREC const &radius_x,
              PREC const &radius_y,
              PREC const &height,
              BaseMaterial **const &material_dev_ptr,
              Transformation *const &trans_dev_ptr)
{
    createCylinder<<<1, 1>>>(dev_ptr,
            center_of_bottom_face,
            radius_x, radius_y,
            height,
            material_dev_ptr,
            trans_dev_ptr);
}

__global__
void GpuCreator::createDisk(BaseGeometry ** dev_ptr,
                Point position,
                Direction norm_vec,
                PREC radius,
                BaseMaterial **material_dev_ptr,
                Transformation *trans_dev_ptr)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        (*dev_ptr)= new BaseDisk(position, norm_vec, radius,
                                 material_dev_ptr == nullptr ? nullptr
                                                             : *material_dev_ptr,
                                 trans_dev_ptr);
    }
}

void GpuCreator::Disk(BaseGeometry ** const &dev_ptr,
          Point const &position,
          Direction const &norm_vec,
          PREC const &radius,
          BaseMaterial **const &material_dev_ptr,
          Transformation *const &trans_dev_ptr)
{
    createDisk<<<1, 1>>>(dev_ptr,
            position, norm_vec, radius,
            material_dev_ptr,
            trans_dev_ptr);
}

__global__
void GpuCreator::createEllipsoid(BaseGeometry ** dev_ptr,
                     Point center,
                     PREC radius_x,
                     PREC radius_y,
                     PREC radius_z,
                     BaseMaterial **material_dev_ptr,
                     Transformation *trans_dev_ptr)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        (*dev_ptr) = new BaseEllipsoid(center,
                                       radius_x, radius_y, radius_z,
                                       material_dev_ptr == nullptr ? nullptr
                                                                   : *material_dev_ptr,
                                       trans_dev_ptr);
    }
}

void GpuCreator::Ellipsoid(BaseGeometry ** const &dev_ptr,
               Point const &center,
               PREC const &radius_x,
               PREC const &radius_y,
               PREC const &radius_z,
               BaseMaterial **const &material_dev_ptr,
               Transformation *const &trans_dev_ptr)
{
    createEllipsoid<<<1, 1>>>(dev_ptr,
            center, radius_x, radius_y, radius_z,
            material_dev_ptr, trans_dev_ptr);
}

__global__
void GpuCreator::createNormalTriangle(BaseGeometry ** dev_ptr,
                          Point vertex1, Direction normal1,
                          Point vertex2, Direction normal2,
                          Point vertex3, Direction normal3,
                          BaseMaterial **material_dev_ptr,
                          Transformation *trans_dev_ptr)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        (*dev_ptr) = new BaseNormalTriangle(vertex1, normal1,
                                            vertex2, normal2,
                                            vertex3, normal3,
                                            material_dev_ptr == nullptr ? nullptr
                                                                        : *material_dev_ptr,
                                            trans_dev_ptr);
    }
}

void GpuCreator::NormalTriangle(BaseGeometry ** const &dev_ptr,
                    Point const &vertex1, Direction const &normal1,
                    Point const &vertex2, Direction const &normal2,
                    Point const &vertex3, Direction const &normal3,
                    BaseMaterial **const &material_dev_ptr,
                    Transformation *const &trans_dev_ptr)
{
    createNormalTriangle<<<1, 1>>>(dev_ptr,
            vertex1, normal1,
            vertex2, normal2,
            vertex3, normal3,
            material_dev_ptr, trans_dev_ptr);
}

__global__
void GpuCreator::createPlane(BaseGeometry ** dev_ptr,
                 Point pos,
                 Direction norm_vec,
                 BaseMaterial **material_dev_ptr,
                 Transformation *trans_dev_ptr)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        (*dev_ptr) = new BasePlane(pos, norm_vec,
                                   material_dev_ptr == nullptr ? nullptr
                                                               : *material_dev_ptr,
                                   trans_dev_ptr);
    }
}

void GpuCreator::Plane(BaseGeometry ** const &dev_ptr,
           Point const &pos,
           Direction const &norm_vec,
           BaseMaterial **const &material_dev_ptr,
           Transformation *const &trans_dev_ptr)
{
    createPlane<<<1, 1>>>(dev_ptr, pos, norm_vec,
            material_dev_ptr, trans_dev_ptr);
}

__global__
void GpuCreator::createQuadric(BaseGeometry ** dev_ptr,
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
                   Transformation *trans_dev_ptr)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        (*dev_ptr) = new BaseQuadric(center,
                                     a, b, c, d, e, f, g, h, i, j,
                                     x0, x1, y0, y1, z0, z1,
                                     material_dev_ptr == nullptr ? nullptr
                                                                 : *material_dev_ptr,
                                     trans_dev_ptr);
    }
}

void GpuCreator::Quadric(BaseGeometry ** const &dev_ptr,
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
             Transformation *const &trans_dev_ptr)
{
    createQuadric<<<1, 1>>>(dev_ptr,
            center,
            a, b, c, d, e, f, g, h, i, j,
            x0, x1, y0, y1, z0, z1,
            material_dev_ptr, trans_dev_ptr);
}

__global__
void GpuCreator::createRing(BaseGeometry ** dev_ptr,
                Point pos,
                Direction norm_vec,
                PREC radius1,
                PREC radius2,
                BaseMaterial **material_dev_ptr,
                Transformation *trans_dev_ptr)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        (*dev_ptr) = new BaseRing(pos, norm_vec,
                                  radius1, radius2,
                                  material_dev_ptr == nullptr ? nullptr
                                                              : *material_dev_ptr,
                                  trans_dev_ptr);
    }
}

void GpuCreator::Ring(BaseGeometry ** const &dev_ptr,
          Point const &pos,
          Direction const &norm_vec,
          PREC const &radius1,
          PREC const &radius2,
          BaseMaterial **const &material_dev_ptr,
          Transformation *const &trans_dev_ptr)
{
    createRing<<<1, 1>>>(dev_ptr,
            pos, norm_vec,
            radius1, radius2,
            material_dev_ptr, trans_dev_ptr);
}

__global__
void GpuCreator::createSphere(BaseGeometry ** dev_ptr,
                  Point pos,
                  PREC radius,
                  BaseMaterial **material_dev_ptr,
                  Transformation *trans_dev_ptr)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        (*dev_ptr) = new BaseSphere(pos, radius,
                                    material_dev_ptr == nullptr ? nullptr
                                                                : *material_dev_ptr,
                                    trans_dev_ptr);
    }
}

void GpuCreator::Sphere(BaseGeometry ** const &dev_ptr,
            Point const &pos,
            PREC const &radius,
            BaseMaterial **const &material_dev_ptr,
            Transformation *const &trans_dev_ptr)
{
    createSphere<<<1, 1>>>(dev_ptr, pos, radius,
            material_dev_ptr, trans_dev_ptr);
}

__global__
void GpuCreator::createTriangle(BaseGeometry ** dev_ptr,
                    Point a,
                    Point b,
                    Point c,
                    BaseMaterial **material_dev_ptr,
                    Transformation *trans_dev_ptr)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        (*dev_ptr) = new BaseTriangle(a, b, c,
                                      material_dev_ptr == nullptr ? nullptr
                                                                  : *material_dev_ptr,
                                      trans_dev_ptr);
    }
}

void GpuCreator::Triangle(BaseGeometry ** const &dev_ptr,
              Point const &a,
              Point const &b,
              Point const &c,
              BaseMaterial **const &material_dev_ptr,
              Transformation *const &trans_dev_ptr)
{
    createTriangle<<<1, 1>>>(dev_ptr, a, b, c,
            material_dev_ptr, trans_dev_ptr);
}

__global__
void GpuCreator::createBoundBox(BaseGeometry **dev_ptr,
                    BaseGeometry ***objs,
                    int n,
                    Transformation *trans_dev_ptr)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        auto tmp = new BaseBoundBox(trans_dev_ptr);
        for (auto i = 0; i < n; ++i)
            tmp->addObj(*(objs[i]));
        (*dev_ptr) = tmp;
    }

}

void GpuCreator::BoundBox(BaseGeometry** const &dev_ptr,
              BaseGeometry *** const &objs,
              int const &n,
              Transformation *const &trans_dev_ptr)
{
    createBoundBox<<<1, 1>>>(dev_ptr, objs, n, trans_dev_ptr);
}

__global__
void GpuCreator::createBVH(BaseGeometry **dev_ptr,
               BaseGeometry ***objs,
               int n)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        auto tmp = new BaseBVH;
        for (auto i = 0; i < n; ++i)
            tmp->addObj(*(objs[i]));
        (*dev_ptr) = tmp;
    }

}

void GpuCreator::BVH(BaseGeometry **const &dev_ptr,
         BaseGeometry ***const &objs,
         int const &n)
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    createBVH<<<1, 1, 0, stream>>>(dev_ptr, objs, n);
    cudaStreamDestroy(stream);
}
