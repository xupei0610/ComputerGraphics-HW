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
class BaseGeometry; // primarily used for gpu
class Geometry; // a cpu object

// TODO Polygon
//class Polygon;
// TODO Torus
//class Torus

// TODO Constructive Solid Geometry
// TODO Implicit Twisted Super Quadric
// TODO Procedurally generated terrain/heightfields
}

class px::Geometry
{
public:
    virtual BaseGeometry **devPtr() = 0;
    virtual void up2Gpu() = 0;
    virtual void clearGpuData() = 0;

    virtual BaseGeometry * const &obj() const noexcept = 0;

protected:
    Geometry() = default;
    ~Geometry() = default;

};

class px::BaseGeometry
{
protected:
    const BaseMaterial * _material;
    const Transformation * _transformation;

    int _n_vertices;
    Point * _raw_vertices;

public:

    PX_CUDA_CALLABLE
    inline const BaseMaterial * const &material() const noexcept
    {
        return _material;
    }

    PX_CUDA_CALLABLE
    inline const Transformation * const &transform() const noexcept
    {
        return _transformation;
    }

    PX_CUDA_CALLABLE
    const BaseGeometry * hit(Ray const &ray,
                       PREC const &range_start,
                       PREC const &range_end,
                       PREC &hit_at) const;
    PX_CUDA_CALLABLE
    Direction normal(PREC const &x,
                     PREC const &y,
                     PREC const &z) const;
    PX_CUDA_CALLABLE
    Vec3<PREC> textureCoord(PREC const &x, PREC const &y, PREC const &z) const;
    PX_CUDA_CALLABLE
    virtual Point * rawVertices(int &n_vertices) const noexcept
    {
        n_vertices = _n_vertices;
        return _raw_vertices;
    }
    PX_CUDA_CALLABLE
    virtual Vec3<PREC> textureCoord(Point const &p) const
    {
        return textureCoord(p.x, p.y, p.z);
    }
    PX_CUDA_CALLABLE
    Direction normVec(Point const &p) const
    {
        return normal(p.x, p.y, p.z);
    }
    PX_CUDA_CALLABLE
    virtual ~BaseGeometry();
protected:
    PX_CUDA_CALLABLE
    virtual Vec3<PREC> getTextureCoord(PREC const &x,
                                         PREC const &y,
                                         PREC const &z) const = 0;
    PX_CUDA_CALLABLE
    inline Vec3<PREC> getTextureCoord(Point const &p)
    {
        return getTextureCoord(p.x, p.y, p.z);
    }
    PX_CUDA_CALLABLE
    virtual const BaseGeometry * hitCheck(Ray const &ray,
                                          PREC const &range_start,
                                          PREC const &range_end,
                                          PREC &hit_at) const = 0;
    PX_CUDA_CALLABLE
    virtual Direction normalVec(PREC const &x, PREC const &y,
                                PREC const &z) const = 0;
    PX_CUDA_CALLABLE
    inline Direction normalVec(Point const &p) const
    {
        return normalVec(p.x, p.y, p.z);
    }

    PX_CUDA_CALLABLE
    BaseGeometry(const BaseMaterial * const &material,
                 const Transformation * const &trans,
                 int const &n_vertices);

private:

    BaseGeometry &operator=(BaseGeometry const &) = delete;
    BaseGeometry &operator=(BaseGeometry &&) = delete;

};

#endif // PX_CG_OBJECT_GEOMETRY_BASE_GEOMETRY_HPP
