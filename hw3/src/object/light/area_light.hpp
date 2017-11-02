#ifndef PX_CG_OBJECT_LIGHT_AREA_LIGHT_HPP
#define PX_CG_OBJECT_LIGHT_AREA_LIGHT_HPP

#include "object/light/base_light.hpp"

namespace px
{
class BaseAreaLight;
class AreaLight;
}

class px::BaseAreaLight
{
public:
    PX_CUDA_CALLABLE
    static PREC attenuate(void * const &obj, PREC const &x, PREC const &y, PREC const &z);
    __device__
    static Direction dirFromDevice(void * const &obj, PREC const &x, PREC const &y, PREC const &z, PREC &dist, curandState_t * const &);
    static Direction dirFromHost(BaseAreaLight * const &obj, PREC const &x, PREC const &y, PREC const &z, PREC &dist);

    void setCenter(Point const &center);
    void setRadius(PREC const &r);

    ~BaseAreaLight() = default;

protected:
    Point _center;
    PREC _radius;
    PREC _radius2;

    BaseAreaLight(Point const &center,
                  PREC const &radius);

    BaseAreaLight &operator=(BaseAreaLight const &) = delete;
    BaseAreaLight &operator=(BaseAreaLight &&) = delete;

    friend class AreaLight;
};

class px::AreaLight : public BaseLight
{
public:
    const static LightType TYPE;

    static std::shared_ptr<BaseLight> create(Light const &light,
                                             Point const &center,
                                             PREC const &radius);

    void up2Gpu() override;
    void clearGpuData() override;

    PREC attenuate(PREC const &x, PREC const &y, PREC const &z) const override;
    Direction dirFrom(PREC const &x, PREC const &y, PREC const &z, PREC &dist) const override;

    void setCenter(Point const &c);
    void setRadius(PREC const &r);

    ~AreaLight();
protected:
    BaseAreaLight *_obj;
    void *_gpu_obj;
    bool _need_upload;

    AreaLight(Light const &light,
              Point const &center,
              PREC const &radius);

    AreaLight &operator=(AreaLight const &) = delete;
    AreaLight &operator=(AreaLight &&) = delete;
};

#endif // PX_CG_OBJECT_LIGHT_AREA_LIGHT_HPP
