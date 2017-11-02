#include "object/light/spot_light.hpp"

#include <cfloat>

using namespace px;

BaseSpotLight::BaseSpotLight(Point const &pos,
                             Direction const &direction,
                             PREC const &half_angle1,
                             PREC const &half_angle2,
                             PREC const &falloff)
        : _pos(pos), _dir(direction), _falloff(falloff)
{
    setAngles(half_angle1, half_angle2);
}


void BaseSpotLight::setPos(Point const &pos)
{
    _pos = pos;
}

void BaseSpotLight::setDir(Direction const &direction)
{
    _dir = direction;
}

void BaseSpotLight::setAngles(PREC const &half_angle1, PREC const &half_angle2)
{
    _inner_ha = half_angle1 < 0 ?
                std::fmod(half_angle1, PREC(PI2)) + PREC(PI2) : std::fmod(half_angle1,PREC(PI2));
    _outer_ha = half_angle2 < 0 ?
                std::fmod(half_angle2, PREC(PI2)) + PREC(PI2) : std::fmod(half_angle2,PREC(PI2));

    if (_inner_ha_cosine > PI)
        _inner_ha_cosine = PI;
    if (_outer_ha_cosine > PI)
        _outer_ha_cosine = PI;

    if (_outer_ha < _inner_ha)
    {
        auto tmp = _inner_ha;
        _inner_ha = _outer_ha;
        _outer_ha = tmp;
    }

    _inner_ha_cosine = std::cos(_inner_ha);
    _outer_ha_cosine = std::cos(_outer_ha);
    _multiplier = 1.0 / (_outer_ha_cosine - _inner_ha_cosine);
}

void BaseSpotLight::setFalloff(PREC const &falloff)
{
    _falloff = falloff;
}

PX_CUDA_CALLABLE
PREC BaseSpotLight::attenuate(void *const &obj,
                                     PREC const &x, PREC const &y, PREC const &z)
{
    auto o = reinterpret_cast<BaseSpotLight*>(obj);
    PREC dx = x-o->_pos.x;
    PREC dy = y-o->_pos.y;
    PREC dz = z-o->_pos.z;
    PREC nrm2 = dx*dx + dy*dy + dz*dz;
    if (nrm2 < EPSILON)
        return FLT_MAX;

    PREC nrm = std::sqrt(nrm2);

    dx /= nrm;
    dy /= nrm;
    dz /= nrm;

    PREC cosine = o->_dir.x * dx + o->_dir.y * dy + o->_dir.z * dz;

    if (cosine >= o->_inner_ha_cosine)
        return 1.0/nrm2;
    if (cosine > o->_outer_ha_cosine)
        return std::pow(((o->_outer_ha_cosine-cosine)*o->_multiplier), o->_falloff)/nrm2;
    return 0;
}

Direction BaseSpotLight::dirFromHost(BaseSpotLight *const &obj,
                                     PREC const &x, PREC const &y,
                                     PREC const &z, PREC &dist)
{
    auto dx = obj->_pos.x - x;
    auto dy = obj->_pos.y - y;
    auto dz = obj->_pos.z - z;

    dist = std::sqrt(dx*dx + dy*dy + dz*dz);
    return {dx, dy, dz};
}

PX_CUDA_CALLABLE
Direction BaseSpotLight::dirFromDevice(void *const &obj, PREC const &x,
                                              PREC const &y, PREC const &z,
                                              PREC &dist,
                                              curandState_t * const &)
{
    auto o = reinterpret_cast<BaseSpotLight*>(obj);

    auto dx = o->_pos.x - x;
    auto dy = o->_pos.y - y;
    auto dz = o->_pos.z - z;

    dist = std::sqrt(dx*dx + dy*dy + dz*dz);
    return {dx, dy, dz};
}

const LightType SpotLight::TYPE = LightType::PointLight;

std::shared_ptr<BaseLight> SpotLight::create(Light const &light,
                                             Point const &pos,
                                             Direction const &direction,
                                             PREC const &half_angle1,
                                             PREC const &half_angle2,
                                             PREC const &falloff)
{
    return std::shared_ptr<BaseLight>(new SpotLight(light, pos, direction,
                                                    half_angle1, half_angle2,
                                                    falloff));
}

SpotLight::SpotLight(Light const &light,
                     Point const &pos,
                     Direction const &direction,
                     PREC const &half_angle1,
                     PREC const &half_angle2,
                     PREC const &falloff)
        : BaseLight(TYPE, light),
          _obj(new BaseSpotLight(pos, direction,
                                 half_angle1, half_angle2,
                                 falloff)),
          _gpu_obj(nullptr),
          _need_upload(true)
{}

SpotLight::~SpotLight()
{
    delete _obj;
#ifdef USE_CUDA
    clearGpuData();
#endif
}

PREC SpotLight::attenuate(PREC const &x, PREC const &y, PREC const &z) const
{
    return BaseSpotLight::attenuate(_obj, x, y, z);
}

Direction SpotLight::dirFrom(PREC const &x, PREC const &y, PREC const &z, PREC &dist) const
{
    return BaseSpotLight::dirFromHost(_obj, x, y, z, dist);
}

void SpotLight::setPos(Point const &pos)
{
    _obj->setPos(pos);
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void SpotLight::setDir(Direction const &direction)
{
    _obj->setDir(direction);
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void SpotLight::setAngles(PREC const &half_angle1, PREC const &half_angle2)
{
    _obj->setAngles(half_angle1, half_angle2);
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void SpotLight::setFalloff(PREC const &falloff)
{
    _obj->setFalloff(falloff);
#ifdef USE_CUDA
    _need_upload = true;
#endif
}

#ifdef USE_CUDA
__device__ fnAttenuate_t __fn_attenuate_spot_light = BaseSpotLight::attenuate;
__device__ fnDirFrom_t  __fn_dir_from_spot_light = BaseSpotLight::dirFromDevice;
#endif

void SpotLight::up2Gpu()
{
#ifdef USE_CUDA
    static fnAttenuate_t fn_attenuate_h = nullptr;
    static fnDirFrom_t  fn_dir_from_h;
    if (_need_upload || BaseLight::need_upload)
    {
        if (dev_ptr == nullptr)
        {
            PX_CUDA_CHECK(cudaMalloc(&_gpu_obj, sizeof(BaseSpotLight)));
            PX_CUDA_CHECK(cudaMalloc(&dev_ptr, sizeof(LightObj)));
        }
        if (fn_attenuate_h == nullptr)
        {
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_attenuate_h, __fn_attenuate_spot_light, sizeof(fnAttenuate_t)));
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_dir_from_h, __fn_dir_from_spot_light, sizeof(fnDirFrom_t)));
        }
        PX_CUDA_CHECK(cudaMemcpy(_gpu_obj, _obj, sizeof(BaseSpotLight),
                                 cudaMemcpyHostToDevice));

        LightObj tmp(_gpu_obj, type, _light, fn_attenuate_h, fn_dir_from_h);

        PX_CUDA_CHECK(cudaMemcpy(dev_ptr, &tmp, sizeof(LightObj),
                                 cudaMemcpyHostToDevice));

        _need_upload = false;
        BaseLight::need_upload = false;
    }
#endif
}

void SpotLight::clearGpuData()
{
#ifdef USE_CUDA
    if (_gpu_obj != nullptr)
    {
        PX_CUDA_CHECK(cudaFree(_gpu_obj));
        _gpu_obj = nullptr;
    }
    BaseLight::clearGpuData();
#endif
}
