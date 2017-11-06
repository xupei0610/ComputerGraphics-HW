#ifndef PX_CG_OBJECT_GEOMETRY_BASE_GEOMETRY_HPP
#define PX_CG_OBJECT_GEOMETRY_BASE_GEOMETRY_HPP

#include <iostream>

#include <unordered_set>
#include <limits>
#include <vector>
#include <list>

#include "object/base_object.hpp"
#include "object/material.hpp"

namespace px
{
class GeometryObj;
typedef GeometryObj * (*fnHit_t)(void * const &, Ray const &, PREC const &, PREC const&, PREC &);
typedef Direction (*fnNormal_t)(void * const &, PREC const &, PREC const &, PREC const&, bool &);
typedef Vec3<PREC> (*fnTextureCoord_t)(void * const &, PREC const &, PREC const &, PREC const&);


class BaseGeometry;

// TODO Polygon
//class Polygon;
// TODO Torus
//class Torus

// TODO Constructive Solid Geometry
// TODO Implicit Twisted Super Quadric
// TODO Procedurally generated terrain/heightfields
}

class px::GeometryObj
{
protected:
    void *obj;
    fnHit_t fn_hit;
    fnNormal_t fn_normal;
    fnTextureCoord_t fn_texture_coord;

    MaterialObj *mat;
    Transformation *trans;
public:
    PX_CUDA_CALLABLE
    inline GeometryObj *hit(Ray const &ray,
                            PREC const &t_start,
                            PREC const &t_end,
                            PREC &hit_at)
    {
        if (trans == nullptr)
            return fn_hit(obj, ray, t_start, t_end, hit_at);
        return fn_hit(obj, {trans->point2ObjCoord(ray.original), trans->direction(ray.direction)},
                      t_start, t_end, hit_at);
    }
    PX_CUDA_CALLABLE
    inline Direction normal(PREC const &x,
                            PREC const &y,
                            PREC const &z,
                            bool &double_face)
    {
        if (trans == nullptr)
            return fn_normal(obj, x, y, z, double_face);
        auto p = trans->point2ObjCoord(x, y, z);
        return trans->normal(fn_normal(obj, p.x, p.y, p.z, double_face));
    }
    PX_CUDA_CALLABLE
    inline Vec3<PREC> textureCoord(PREC const &x,
                                   PREC const &y,
                                   PREC const &z)
    {
        if (trans == nullptr)
            return fn_texture_coord(obj, x, y, z);
        auto p = trans->point2ObjCoord(x, y, z);
        return fn_texture_coord(obj, p.x, p.y, p.z);
    }
    PX_CUDA_CALLABLE
    inline Direction normal(Point const &p, bool &double_face)
    {
        return normal(p.x, p.y, p.z, double_face);
    }
    PX_CUDA_CALLABLE
    inline Vec3<PREC> textureCoord(Point const &p)
    {
        return textureCoord(p.x, p.y, p.z);
    }
    PX_CUDA_CALLABLE
    inline MaterialObj * const &material() const noexcept
    {
        return mat;
    }

    GeometryObj(void *obj,
                fnHit_t const &fn_hit,
                fnNormal_t const &fn_normal,
                fnTextureCoord_t const &fn_texture_coord,
                MaterialObj *const &mat,
                Transformation *const &trans);
    ~GeometryObj() = default;
protected:
    GeometryObj &operator=(GeometryObj const &) = delete;
    GeometryObj &operator=(GeometryObj &&) = delete;
};


class px::BaseGeometry
{
protected:
    std::shared_ptr<BaseMaterial> mat;
    std::shared_ptr<Transformation> trans;

    int n_vertices;
    Point * raw_vertices;

    GeometryObj *dev_ptr;

public:
    inline GeometryObj *devPtr() const noexcept { return dev_ptr; }
    virtual void up2Gpu() = 0;
    virtual void clearGpuData();

    inline const BaseGeometry * hit(Ray const &ray,
                             PREC const &t_start,
                             PREC const &t_end,
                             PREC &hit_at) const
    {
        if (trans == nullptr)
            return hitCheck(ray, t_start, t_end, hit_at);
        return hitCheck({trans->point2ObjCoord(ray.original), trans->direction(ray.direction)},
                        t_start, t_end, hit_at);
    }
    inline Direction normal(PREC const &x,
                     PREC const &y,
                     PREC const &z,
                            bool &double_face) const
    {
        if (trans == nullptr)
            return normalVec(x, y, z, double_face);
        auto p = trans->point2ObjCoord(x, y, z);
        return trans->normal(normalVec(p.x, p.y, p.z, double_face));
    }

    inline Vec3<PREC> textureCoord(PREC const &x, PREC const &y, PREC const &z) const
    {
        if (trans == nullptr)
            return getTextureCoord(x, y, z);
        auto p = trans->point2ObjCoord(x, y, z);
        return getTextureCoord(p.x, p.y, p.z);
    }

    inline Direction normal(Point const &p, bool &double_face) const
    {
        return normal(p.x, p.y, p.z, double_face);
    }
    inline Vec3<PREC> textureCoord(Point const &p) const
    {
        return textureCoord(p.x, p.y, p.z);
    }

    inline std::shared_ptr<BaseMaterial> const &material() const noexcept
    {
        return mat;
    }

    Point * rawVertices(int &n_vertices) const noexcept;
    void resetVertices(int const &n_vertices);

    inline std::shared_ptr<Transformation> const & transform()
    {
        return trans;
    }

protected:
    virtual Vec3<PREC> getTextureCoord(PREC const &x,
                                       PREC const &y,
                                       PREC const &z) const = 0;
    virtual const BaseGeometry * hitCheck(Ray const &ray,
                                          PREC const &range_start,
                                          PREC const &range_end,
                                          PREC &hit_at) const = 0;
    virtual Direction normalVec(PREC const &x,
                                PREC const &y,
                                PREC const &z,
                                bool &double_face) const = 0;

    BaseGeometry(std::shared_ptr<BaseMaterial> const &material,
                 std::shared_ptr<Transformation> const &trans,
                 int const &n_vertices);
    virtual ~BaseGeometry();
    BaseGeometry &operator=(BaseGeometry const &) = delete;
    BaseGeometry &operator=(BaseGeometry &&) = delete;

    friend class Structure;
};

#endif // PX_CG_OBJECT_GEOMETRY_BASE_GEOMETRY_HPP
