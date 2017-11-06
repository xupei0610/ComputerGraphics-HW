#ifndef PX_CG_MATERIAL_BASE_MATERIAL_HPP
#define PX_CG_MATERIAL_BASE_MATERIAL_HPP

#include "object/base_object.hpp"

#include <vector>
#include <cstdint>
#include <memory>

namespace px
{

class MaterialObj;
typedef Light (*fnAmbient_t)(void * const &, PREC const &, PREC const &, PREC const &);
typedef fnAmbient_t fnDiffuse_t;
typedef fnAmbient_t fnSpecular_t;
typedef fnAmbient_t fnTransmissive_t;
typedef PREC (*fnShininess_t)(void * const &, PREC const &, PREC const &, PREC const &);
typedef PREC (*fnRefractiveIndex_t)(void * const &, PREC const &, PREC const &, PREC const &);

class BaseMaterial;

// TODO Wood
//class Wood;
// TODO Marble
//class Marble;
// TODO mandelbrot set
// TODO Procedural Bump mapping

}

class px::MaterialObj
{
protected:
    void *obj;
    fnAmbient_t fn_ambient;
    fnDiffuse_t fn_diffuse;
    fnSpecular_t fn_specular;
    fnShininess_t fn_shininess;
    fnTransmissive_t fn_transmissive;
    fnRefractiveIndex_t fn_refractive_index;

public:
    PX_CUDA_CALLABLE
    inline Light ambient(PREC const &u, PREC const &v, PREC const &w)
    {
        return fn_ambient(obj, u, v, w);
    }
    PX_CUDA_CALLABLE
    inline Light diffuse(PREC const &u, PREC const &v, PREC const &w)
    {
        return fn_diffuse(obj, u, v, w);
    }
    PX_CUDA_CALLABLE
    inline Light specular(PREC const &u, PREC const &v, PREC const &w)
    {
        return fn_specular(obj, u, v, w);
    }
    PX_CUDA_CALLABLE
    inline Light transmissive(PREC const &u, PREC const &v, PREC const &w)
    {
        return fn_transmissive(obj, u, v, w);
    }
    PX_CUDA_CALLABLE
    inline PREC Shininess(PREC const &u, PREC const &v, PREC const &w)
    {
        return fn_shininess(obj, u, v, w);
    }
    PX_CUDA_CALLABLE
    inline PREC refractiveIndex(PREC const &u, PREC const &v, PREC const &w)
    {
        return fn_refractive_index(obj, u, v, w);
    }
    PX_CUDA_CALLABLE
    inline Light ambient(Point const &p)
    {
        return fn_ambient(obj, p.x, p.y, p.z);
    }
    PX_CUDA_CALLABLE
    inline Light diffuse(Point const &p)
    {
        return fn_diffuse(obj, p.x, p.y, p.z);
    }
    PX_CUDA_CALLABLE
    inline Light specular(Point const &p)
    {
        return fn_specular(obj, p.x, p.y, p.z);
    }
    PX_CUDA_CALLABLE
    inline Light transmissive(Point const &p)
    {
        return fn_transmissive(obj, p.x, p.y, p.z);
    }
    PX_CUDA_CALLABLE
    inline PREC Shininess(Point const &p)
    {
        return fn_shininess(obj, p.x, p.y, p.z);
    }
    PX_CUDA_CALLABLE
    inline PREC refractiveIndex(Point const &p)
    {
        return fn_refractive_index(obj, p.x, p.y, p.z);
    }

    MaterialObj(void * obj,
                fnAmbient_t const &fn_ambient, fnDiffuse_t const &fn_diffuse,
                fnSpecular_t const &fn_specular, fnShininess_t const &fn_shininess,
                fnTransmissive_t const &fn_transmissive, fnRefractiveIndex_t const &fn_refractive_index);
    ~MaterialObj() = default;

    MaterialObj &operator=(MaterialObj const &) = delete;
    MaterialObj &operator=(MaterialObj &&) = delete;

};

class px::BaseMaterial
{
protected:
    MaterialObj *dev_ptr;
public:
    inline MaterialObj* devPtr() const noexcept { return dev_ptr; }
    virtual void up2Gpu() = 0;
    virtual void clearGpuData();

    virtual PREC Shininess(PREC const &u, PREC const &v, PREC const &w) const = 0;
    virtual PREC refractiveIndex(PREC const &u, PREC const &v, PREC const &w) const = 0;

    inline Light ambient(PREC const &u, PREC const &v, PREC const &w) const
    {
        return getAmbient(u, v, w);
    }
    inline Light diffuse(PREC const &u, PREC const &v, PREC const &w) const
    {
        return getDiffuse(u, v, w);
    }
    inline Light specular(PREC const &u, PREC const &v, PREC const &w) const
    {
        return getSpecular(u, v, w);
    }
    inline Light transmissive(PREC const &u, PREC const &v, PREC const &w) const
    {
        return getTransmissive(u, v, w);
    }
    inline Light ambient(Point const &p) const
    {
        return ambient(p.x, p.y, p.z);
    }

    inline Light diffuse(Point const &p) const
    {
        return diffuse(p.x, p.y, p.z);
    }

    inline Light specular(Point const &p) const
    {
        return specular(p.x, p.y, p.z);
    }

    inline Light transmissive(Point const &p) const
    {
        return transmissive(p.x, p.y, p.z);
    }

    inline PREC Shininess(Point const &p) const
    {
        return Shininess(p.x, p.y, p.z);
    }

    inline PREC refractiveIndex(Point const &p) const
    {
        return refractiveIndex(p.x, p.y, p.z);
    }

protected:
    virtual Light getAmbient(PREC const &u, PREC const &v, PREC const &w) const = 0;
    virtual Light getDiffuse(PREC const &u, PREC const &v, PREC const &w) const = 0;
    virtual Light getSpecular(PREC const &u, PREC const &v, PREC const &w) const = 0;
    virtual Light getTransmissive(PREC const &u, PREC const &v, PREC const &w) const = 0;

    BaseMaterial();
    virtual ~BaseMaterial() = default;

    BaseMaterial &operator=(BaseMaterial const &) = delete;
    BaseMaterial &operator=(BaseMaterial &&) = delete;
};

#endif // PX_CG_MATERIAL_BASE_MATERIAL_HPP
