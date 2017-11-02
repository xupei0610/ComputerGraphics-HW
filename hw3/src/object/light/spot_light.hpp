#ifndef PX_CG_OBJECT_LIGHT_SPOT_LIGHT_HPP
#define PX_CG_OBJECT_LIGHT_SPOT_LIGHT_HPP

#include "object/light/base_light.hpp"

namespace px
{
class BaseSpotLight;
class SpotLight;
}

class px::BaseSpotLight
{
public:
    PX_CUDA_CALLABLE
    static PREC attenuate(void * const &obj, PREC const &x, PREC const &y, PREC const &z);
    PX_CUDA_CALLABLE
    static Direction dirFromDevice(void * const &obj, PREC const &x, PREC const &y, PREC const &z, PREC &dist, curandState_t * const &);
    static Direction dirFromHost(BaseSpotLight * const &obj, PREC const &x, PREC const &y, PREC const &z, PREC &dist);

    void setPos(Point const &pos);
    void setDir(Direction const &direction);
    void setAngles(PREC const &half_angle1, PREC const &half_angle2);
    void setFalloff(PREC const &falloff);

    ~BaseSpotLight() = default;

protected:
    Point _pos;
    Direction _dir;
    PREC _inner_ha;
    PREC _outer_ha;
    PREC _falloff;
    PREC _inner_ha_cosine;
    PREC _outer_ha_cosine;
    PREC _multiplier; // 1.0 / (_outer_ha_cosine - inner_ha_cosine)

    BaseSpotLight(Point const &pos,
                  Direction const &direction,
                  PREC const &half_angle1,
                  PREC const &half_angle2,
                  PREC const &falloff);

    BaseSpotLight &operator=(BaseSpotLight const &) = delete;
    BaseSpotLight &operator=(BaseSpotLight &&) = delete;

    friend class SpotLight;
};

class px::SpotLight : public BaseLight
{
public:
    const static LightType TYPE;

    static std::shared_ptr<BaseLight> create(Light const &light,
                                             Point const &pos,
                                             Direction const &direction,
                                             PREC const &half_angle1,
                                             PREC const &half_angle2,
                                             PREC const &falloff);

    void up2Gpu() override;
    void clearGpuData() override;

    PREC attenuate(PREC const &x, PREC const &y, PREC const &z) const override;
    Direction dirFrom(PREC const &x, PREC const &y, PREC const &z, PREC &dist) const override;

    void setPos(Point const &pos);
    void setDir(Direction const &direction);
    void setAngles(PREC const &half_angle1, PREC const &half_angle2);
    void setFalloff(PREC const &falloff);

    ~SpotLight();
protected:
    BaseSpotLight *_obj;
    void *_gpu_obj;
    bool _need_upload;

    SpotLight(Light const &light,
              Point const &pos,
              Direction const &direction,
              PREC const &half_angle1,
              PREC const &half_angle2,
              PREC const &falloff);

    SpotLight &operator=(SpotLight const &) = delete;
    SpotLight &operator=(SpotLight &&) = delete;
};

#endif // PX_CG_OBJECT_LIGHT_SPOT_LIGHT_HPP
