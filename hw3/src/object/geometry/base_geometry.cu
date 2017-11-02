#include "object/geometry/base_geometry.hpp"

using namespace px;

GeometryObj::GeometryObj(void *obj,
            fnHit_t const &fn_hit,
            fnNormal_t const &fn_normal,
            fnTextureCoord_t const &fn_texture_coord,
            MaterialObj *const &mat,
            Transformation *const &trans)
        : obj(obj), fn_hit(fn_hit), fn_normal(fn_normal), fn_texture_coord(fn_texture_coord),
          mat(mat), trans(trans)
{}

BaseGeometry::BaseGeometry(std::shared_ptr<BaseMaterial> const &material,
                           std::shared_ptr<Transformation> const &trans,
                           int const &n_vertices)
        : mat(material),
          trans(trans),
          n_vertices(n_vertices),
          raw_vertices(n_vertices > 0 ? (new Point[n_vertices]) : nullptr),
          dev_ptr(nullptr)
{
}

BaseGeometry::~BaseGeometry()
{
    delete [] raw_vertices;
}

const BaseGeometry *BaseGeometry::hit(Ray const &ray,
                                PREC const &t_start,
                                PREC const &t_end,
                                PREC &hit_at) const
{
    if (trans == nullptr)
        return hitCheck(ray, t_start, t_end, hit_at);
    return hitCheck({trans->point2ObjCoord(ray.original), trans->direction(ray.direction)},
                    t_start, t_end, hit_at);
}

Direction BaseGeometry::normal(PREC const &x,
                               PREC const &y,
                               PREC const &z) const
{
    if (trans == nullptr)
        return normalVec(x, y, z);
    auto p = trans->point2ObjCoord(x, y, z);
    return trans->normal(normalVec(p.x, p.y, p.z));
}

Vec3<PREC> BaseGeometry::textureCoord(PREC const &x,
                                      PREC const &y,
                                      PREC const &z) const
{
    if (trans == nullptr)
        return getTextureCoord(x, y, z);
    auto p = trans->point2ObjCoord(x, y, z);
    return getTextureCoord(p.x, p.y, p.z);
}

void BaseGeometry::clearGpuData()
{
#ifdef USE_CUDA
    if (trans.use_count() == 1)
        trans->clearGpuData();
    if (mat.use_count() == 1)
        mat->clearGpuData();
    if (dev_ptr == nullptr)
        return;

    PX_CUDA_CHECK(cudaFree(dev_ptr));
    dev_ptr = nullptr;
#endif
}

Point* BaseGeometry::rawVertices(int &n) const noexcept
{
    n = n_vertices;
    return raw_vertices;
}

void BaseGeometry::resetVertices(int const &n)
{
    n_vertices = n > 0 ? n : 0;
    delete [] raw_vertices;
    raw_vertices = n > 0 ? (new Point[n]) : nullptr;
}
