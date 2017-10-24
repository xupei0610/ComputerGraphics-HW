#ifndef PX_CG_MATERIAL_BASE_MATERIAL_HPP
#define PX_CG_MATERIAL_BASE_MATERIAL_HPP

#include "object/base_object.hpp"

#include <vector>
#include <cstdint>
#include <memory>

namespace px
{

class BaseMaterial;
class BumpMapping;

// TODO Wood
//class Wood;
// TODO Marble
//class Marble;
// TODO mandelbrot set
// TODO Procedural Bump mapping

}

// TODO Bump Mapping
class px::BumpMapping
{
public:
    PX_CUDA_CALLABLE
    Light color(double const &u, double const &v) const
    { return {0,0,0}; }

    BumpMapping *up2Gpu() { return nullptr; }
    void clearGpu() {}
    BumpMapping() = default;
    ~BumpMapping() {clearGpu();};
};

class px::BaseMaterial
{
public:
    virtual BaseMaterial *up2Gpu() {return nullptr;}
    virtual void clearGpuData() {}

    PX_CUDA_CALLABLE
    inline Light ambient(double const &u, double const &v, double const &w) const
    {
        if (_bump_mapping == nullptr)
            return getAmbient(u, v, w);
        return _bump_mapping->color(u, v) * getAmbient(u, v, w);
    }
    PX_CUDA_CALLABLE
    inline Light diffuse(double const &u, double const &v, double const &w) const
    {
        if (_bump_mapping == nullptr)
            return getDiffuse(u, v, w);
        return _bump_mapping->color(u, v) * getDiffuse(u, v, w);
    }
    PX_CUDA_CALLABLE
    inline Light specular(double const &u, double const &v, double const &w) const
    {
        if (_bump_mapping == nullptr)
            return getSpecular(u, v, w);
        return _bump_mapping->color(u, v) * getSpecular(u, v, w);
    }
    PX_CUDA_CALLABLE
    inline Light transmissive(double const &u, double const &v, double const &w) const
    {
        if (_bump_mapping == nullptr)
            return getTransmissive(u, v, w);
        return _bump_mapping->color(u, v) * getTransmissive(u, v, w);
    }
    PX_CUDA_CALLABLE
    virtual int specularExp(double const &u, double const &v, double const &w) const = 0;
    PX_CUDA_CALLABLE
    virtual double refractiveIndex(double const &u, double const &v, double const &w) const = 0;
    PX_CUDA_CALLABLE
    inline Light ambient(Point const &p) const
    {
        return ambient(p.x, p.y, p.z);
    }
    PX_CUDA_CALLABLE
    inline Light diffuse(Point const &p) const
    {
        return diffuse(p.x, p.y, p.z);
    }
    PX_CUDA_CALLABLE
    inline Light specular(Point const &p) const
    {
        return specular(p.x, p.y, p.z);
    }
    PX_CUDA_CALLABLE
    inline Light transmissive(Point const &p) const
    {
        return transmissive(p.x, p.y, p.z);
    }
    PX_CUDA_CALLABLE
    inline int specularExp(Point const &p) const
    {
        return specularExp(p.x, p.y, p.z);
    }
    PX_CUDA_CALLABLE
    inline double refractiveIndex(Point const &p) const
    {
        return refractiveIndex(p.x, p.y, p.z);
    }

    virtual ~BaseMaterial() = default;
protected:
    const BumpMapping * _bump_mapping;

    PX_CUDA_CALLABLE
    virtual Light getAmbient(double const &u, double const &v, double const &w) const = 0;
    PX_CUDA_CALLABLE
    virtual Light getDiffuse(double const &u, double const &v, double const &w) const = 0;
    PX_CUDA_CALLABLE
    virtual Light getSpecular(double const &u, double const &v, double const &w) const = 0;
    PX_CUDA_CALLABLE
    virtual Light getTransmissive(double const &u, double const &v, double const &w) const = 0;

    BaseMaterial(const BumpMapping * const &bump_mapping);

    BaseMaterial &operator=(BaseMaterial const &) = delete;
    BaseMaterial &operator=(BaseMaterial &&) = delete;

};

#endif // PX_CG_MATERIAL_BASE_MATERIAL_HPP
