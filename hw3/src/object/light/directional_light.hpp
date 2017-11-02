#ifndef PX_CG_OBJECT_LIGHT_DIRECTIONAL_LIGHT_HPP
#define PX_CG_OBJECT_LIGHT_DIRECTIONAL_LIGHT_HPP

#include "object/light/base_light.hpp"

namespace px
{
class BaseDirectionalLight;
class DirectionalLight;
}

class px::BaseDirectionalLight
{
public:
    PX_CUDA_CALLABLE
    static PREC attenuate(void * const &obj, PREC const &x, PREC const &y, PREC const &z);
    PX_CUDA_CALLABLE
    static Direction dirFromDevice(void * const &obj, PREC const &x, PREC const &y, PREC const &z, PREC &dist, curandState_t * const &);
    static Direction dirFromHost(BaseDirectionalLight * const &obj, PREC const &x, PREC const &y, PREC const &z, PREC &dist);

    void setDir(Direction const &dir);

    ~BaseDirectionalLight() = default;

protected:
    Direction _dir;
    Direction _neg_dir;

    BaseDirectionalLight(Direction const &dir);

    BaseDirectionalLight &operator=(BaseDirectionalLight const &) = delete;
    BaseDirectionalLight &operator=(BaseDirectionalLight &&) = delete;

    friend class DirectionalLight;
};

class px::DirectionalLight : public BaseLight
{
public:
    const static LightType TYPE;

    static std::shared_ptr<BaseLight> create(Light const &light, Direction const &dir);

    void up2Gpu() override;
    void clearGpuData() override;

    PREC attenuate(PREC const &x, PREC const &y, PREC const &z) const override;
    Direction dirFrom(PREC const &x, PREC const &y, PREC const &z, PREC &dist) const override;

    void setDir(Direction const &dir);

    ~DirectionalLight();
protected:
    BaseDirectionalLight *_obj;
    void *_gpu_obj;
    bool _need_upload;

    DirectionalLight(Light const &light, Direction const &dir);

    DirectionalLight &operator=(DirectionalLight const &) = delete;
    DirectionalLight &operator=(DirectionalLight &&) = delete;
};

#endif // PX_CG_OBJECT_LIGHT_DIRECTIONAL_LIGHT_HPP
