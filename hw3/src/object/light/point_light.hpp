#ifndef PX_CG_OBJECT_LIGHT_POINT_LIGHT_HPP
#define PX_CG_OBJECT_LIGHT_POINT_LIGHT_HPP

#include "object/light/base_light.hpp"

namespace px
{
class BasePointLight;
class PointLight;
}

class px::BasePointLight
{
public:
    PX_CUDA_CALLABLE
    static PREC attenuate(void * const &obj, PREC const &x, PREC const &y, PREC const &z);
    PX_CUDA_CALLABLE
    static Direction dirFromDevice(void * const &obj, PREC const &x, PREC const &y, PREC const &z, PREC &dist, curandState_t * const &);
    static Direction dirFromHost(BasePointLight * const &obj, PREC const &x, PREC const &y, PREC const &z, PREC &dist);

    void setPos(Point const &pos);

    ~BasePointLight() = default;

protected:
    Point _pos;

    BasePointLight(Point const &pos);

    BasePointLight &operator=(BasePointLight const &) = delete;
    BasePointLight &operator=(BasePointLight &&) = delete;

    friend class PointLight;
};

class px::PointLight : public BaseLight
{
public:
    const static LightType TYPE;

    static std::shared_ptr<BaseLight> create(Light const &light, Point const &pos);

    void up2Gpu() override;
    void clearGpuData() override;

    PREC attenuate(PREC const &x, PREC const &y, PREC const &z) const override;
    Direction dirFrom(PREC const &x, PREC const &y, PREC const &z, PREC &dist) const override;

    void setPos(Point const &pos);

    ~PointLight();
protected:
    BasePointLight *_obj;
    void *_gpu_obj;
    bool _need_upload;

    PointLight(Light const &light, Point const &pos);

    PointLight &operator=(PointLight const &) = delete;
    PointLight &operator=(PointLight &&) = delete;
};

#endif // PX_CG_OBJECT_LIGHT_POINT_LIGHT_HPP
