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
class BaseGeometry;
class Structure;
// TODO Polygon
//class Polygon;
// TODO Torus
//class Torus

// TODO Constructive Solid Geometry
// TODO Implicit Twisted Super Quadric
// TODO Procedurally generated terrain/heightfields
}

class px::BaseGeometry
{
protected:
    const BaseMaterial * _material;
    const Transformation * _transformation;

    int _n_vertices;
    Point * _raw_vertices;

    friend Structure;

public:
    virtual BaseGeometry *up2Gpu() = 0;
    virtual void clearGpuData() = 0;

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
                       double const &range_start,
                       double const &range_end,
                       double &hit_at) const;
    PX_CUDA_CALLABLE
    Direction normal(double const &x,
                     double const &y,
                     double const &z) const;
    PX_CUDA_CALLABLE
    Vec3<double> textureCoord(double const &x, double const &y, double const &z) const;

    virtual Point * rawVertices(int &n_vertices) const noexcept
    {
        n_vertices = _n_vertices;
        return _raw_vertices;
    }

    PX_CUDA_CALLABLE
    virtual Light ambient(double const &x,
                          double const &y,
                          double const &z) const
    {
        return _material->ambient(textureCoord(x, y, z));
    }
    PX_CUDA_CALLABLE
    virtual Light diffuse(double const &x,
                          double const &y,
                          double const &z) const
    {
        return _material->diffuse(textureCoord(x, y, z));
    }
    PX_CUDA_CALLABLE
    virtual Light specular(double const &x,
                           double const &y,
                           double const &z) const
    {
        return _material->specular(textureCoord(x, y, z));
    }
    PX_CUDA_CALLABLE
    virtual Light transmissive(double const &x,
                               double const &y,
                               double const &z) const
    {
        return _material->transmissive(textureCoord(x, y, z));
    }
    PX_CUDA_CALLABLE
    virtual Vec3<double> textureCoord(Point const &p) const
    {
        return textureCoord(p.x, p.y, p.z);
    }
    PX_CUDA_CALLABLE
    Direction normVec(Point const &p) const
    {
        return normal(p.x, p.y, p.z);
    }
    PX_CUDA_CALLABLE
    virtual Light ambient(Point const &p) const
    {
        return ambient(p.x, p.y, p.z);
    }
    PX_CUDA_CALLABLE
    virtual Light diffuse(Point const &p) const
    {
        return diffuse(p.x, p.y, p.z);
    }
    PX_CUDA_CALLABLE
    virtual Light specular(Point const &p) const
    {
        return specular(p.x, p.y, p.z);
    }
    PX_CUDA_CALLABLE
    virtual Light transmissive(Point const &p) const
    {
        return transmissive(p.x, p.y, p.z);
    }
    PX_CUDA_CALLABLE
    virtual double refractiveIndex(Point const &p) const
    {
        return _material->refractiveIndex(p.x, p.y, p.z);
    }

    ~BaseGeometry();
protected:
    PX_CUDA_CALLABLE
    virtual Vec3<double> getTextureCoord(double const &x,
                                         double const &y,
                                         double const &z) const = 0;
    PX_CUDA_CALLABLE
    inline Vec3<double> getTextureCoord(Point const &p)
    {
        return getTextureCoord(p.x, p.y, p.z);
    }
    PX_CUDA_CALLABLE
    virtual const BaseGeometry * hitCheck(Ray const &ray,
                                          double const &range_start,
                                          double const &range_end,
                                          double &hit_at) const = 0;
    PX_CUDA_CALLABLE
    virtual Direction normalVec(double const &x, double const &y,
                                double const &z) const = 0;
    PX_CUDA_CALLABLE
    inline Direction normalVec(Point const &p) const
    {
        return normalVec(p.x, p.y, p.z);
    }

    BaseGeometry(const BaseMaterial * const &material,
                 const Transformation * const &trans,
                 int const &n_vertices);

private:

    BaseGeometry &operator=(BaseGeometry const &) = delete;
    BaseGeometry &operator=(BaseGeometry &&) = delete;

};

#endif // PX_CG_OBJECT_GEOMETRY_BASE_GEOMETRY_HPP
