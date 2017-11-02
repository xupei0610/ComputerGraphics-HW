#ifndef PX_CG_OBJECT_LIGHT_BASE_LIGHT_HPP
#define PX_CG_OBJECT_LIGHT_BASE_LIGHT_HPP

#include "object/base_object.hpp"

namespace px
{

enum class LightType
{
    PointLight,
    DirectionalLight,
    AreaLight
};

typedef PREC (*fnAttenuate_t)(void * const &, PREC const &, PREC const &, PREC const &);
typedef Direction (*fnDirFrom_t)(void * const &, PREC const &, PREC const &, PREC const &, PREC &, curandState_t * const &);

class LightObj;
class BaseLight;

// TODO Image-based Lighting (environment maps)
}

class px::LightObj
{
protected:
    void *obj;
    fnAttenuate_t fn_attenuate;
    fnDirFrom_t fn_dir_from;
    
public:
    LightType const type;
    Light const light;

    PX_CUDA_CALLABLE
    inline PREC attenuate(PREC const &x, PREC const &y, PREC const &z) const
    {
        return fn_attenuate(obj, x, y, z);
    }
    PX_CUDA_CALLABLE
    inline PREC attenuate(Point const &p) const
    {
        return fn_attenuate(obj, p.x, p.y, p.z);
    }
    PX_CUDA_CALLABLE
    inline Direction dirFrom(PREC const &x, PREC const &y, PREC const &z, PREC &dist, curandState_t * const &state) const
    {
        return fn_dir_from(obj, x, y, z, dist, state);
    }
    PX_CUDA_CALLABLE
    inline Direction dirFrom(Point const &p, PREC &dist, curandState_t * const &state) const
    {
        return fn_dir_from(obj, p.x, p.y, p.z, dist, state);
    }

    LightObj(void * obj,
             LightType const &type, Light const &light,
             fnAttenuate_t const &fn_attenuate, fnDirFrom_t const &fn_dir_from);
    ~LightObj() = default;
    
    LightObj &operator=(BaseLight const &) = delete;
    LightObj &operator=(BaseLight &&) = delete;
};

class px::BaseLight
{
protected:
    Light _light;

    LightObj *dev_ptr;
    bool need_upload;

public:
    LightType const type;

    inline LightObj *devPtr() const noexcept { return dev_ptr; }
    virtual void up2Gpu() = 0;
    virtual void clearGpuData();

    inline Light light() const noexcept { return _light; }
    
    virtual PREC attenuate(PREC const &x, PREC const &y, PREC const &z) const = 0;
    virtual Direction dirFrom(PREC const &x, PREC const &y, PREC const &z, PREC &dist) const = 0;
    
    inline PREC attenuate(Point const &p) { return attenuate(p.x, p.y, p.z); }
    inline Direction dirFrom(Point const &p, PREC &dist) { return dirFrom(p.x, p.y, p.z, dist); }

    void setLight(Light const &light);
    void setLight(PREC const &r, PREC const &g, PREC const &b);

protected:
    BaseLight(LightType const &type, Light const &light);
    virtual ~BaseLight() = default;
    BaseLight &operator=(BaseLight const &) = delete;
    BaseLight &operator=(BaseLight &&) = delete;
};

#endif // PX_CG_OBJECT_LIGHT_BASE_LIGHT_HPP
