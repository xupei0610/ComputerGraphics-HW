#ifndef PX_CG_MATERIAL_BASE_MATERIAL_HPP
#define PX_CG_MATERIAL_BASE_MATERIAL_HPP

#include "object/base_object.hpp"

#include <vector>
#include <cstdint>
#include <memory>

namespace px
{
class Material;
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
protected:
    BumpMapping * _dev_ptr;
    bool _need_update;

public:

    PX_CUDA_CALLABLE
    Light color(PREC const &u, PREC const &v) const;

    void up2Gpu();
    void clearGpuData();
    BumpMapping * devPtr() { return _dev_ptr; }

    BumpMapping();
    ~BumpMapping();
};

class px::Material
{
public:
    virtual BaseMaterial *const &obj() const noexcept = 0;
    virtual BaseMaterial** devPtr() = 0;
    virtual void up2Gpu() = 0;
    virtual void clearGpuData() = 0;

protected:
    Material() = default;
    ~Material() = default;
};

class px::BaseMaterial
{
public:

    PX_CUDA_CALLABLE
    inline Light ambient(PREC const &u, PREC const &v, PREC const &w) const
    {
        if (_bump_mapping == nullptr)
            return getAmbient(u, v, w);
        return _bump_mapping->color(u, v) * getAmbient(u, v, w);
    }
    PX_CUDA_CALLABLE
    inline Light diffuse(PREC const &u, PREC const &v, PREC const &w) const
    {
        if (_bump_mapping == nullptr)
            return getDiffuse(u, v, w);
        return _bump_mapping->color(u, v) * getDiffuse(u, v, w);
    }
    PX_CUDA_CALLABLE
    inline Light specular(PREC const &u, PREC const &v, PREC const &w) const
    {
        if (_bump_mapping == nullptr)
            return getSpecular(u, v, w);
        return _bump_mapping->color(u, v) * getSpecular(u, v, w);
    }
    PX_CUDA_CALLABLE
    inline Light transmissive(PREC const &u, PREC const &v, PREC const &w) const
    {
        if (_bump_mapping == nullptr)
            return getTransmissive(u, v, w);
        return _bump_mapping->color(u, v) * getTransmissive(u, v, w);
    }
    PX_CUDA_CALLABLE
    virtual int specularExp(PREC const &u, PREC const &v, PREC const &w) const = 0;
    PX_CUDA_CALLABLE
    virtual PREC refractiveIndex(PREC const &u, PREC const &v, PREC const &w) const = 0;
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
    inline PREC refractiveIndex(Point const &p) const
    {
        return refractiveIndex(p.x, p.y, p.z);
    }
    PX_CUDA_CALLABLE
    virtual ~BaseMaterial() = default;

    void setBumpMapping(const BumpMapping * const &bm)
    {
        _bump_mapping = bm;
    }

protected:
    const BumpMapping * _bump_mapping;

    PX_CUDA_CALLABLE
    virtual Light getAmbient(PREC const &u, PREC const &v, PREC const &w) const = 0;
    PX_CUDA_CALLABLE
    virtual Light getDiffuse(PREC const &u, PREC const &v, PREC const &w) const = 0;
    PX_CUDA_CALLABLE
    virtual Light getSpecular(PREC const &u, PREC const &v, PREC const &w) const = 0;
    PX_CUDA_CALLABLE
    virtual Light getTransmissive(PREC const &u, PREC const &v, PREC const &w) const = 0;

    PX_CUDA_CALLABLE
    BaseMaterial(const BumpMapping * const &bump_mapping);

    BaseMaterial &operator=(BaseMaterial const &) = delete;
    BaseMaterial &operator=(BaseMaterial &&) = delete;

};

#endif // PX_CG_MATERIAL_BASE_MATERIAL_HPP
