#include <cfloat>
#include "object/light/light.hpp"
#ifdef USE_CUDA
#  include "gpu_creator.hpp"
#endif

using namespace px;

PX_CUDA_CALLABLE
BaseLight::BaseLight(Light const &light)
        : dev_ptr(nullptr), _light(light), need_upload(true)
{}

void BaseLight::setLight(Light const &light)
{
    _light = light;
#ifdef USE_CUDA
    need_upload = true;
#endif
}

std::shared_ptr<BaseLight> DirectionalLight::create(Light const &light,
                                                    Direction const &dir)
{
    return std::shared_ptr<BaseLight>(new DirectionalLight(light, dir));
}

DirectionalLight::DirectionalLight(Light const &light, Direction const &dir)
        : BaseLight(light),
          TYPE(Type::DirectionalLight),
          _dir(dir),
          _neg_dir(dir * -1),
          _need_upload(true)
{}

PX_CUDA_CALLABLE
PREC DirectionalLight::attenuate(PREC const &x,
                                   PREC const &y,
                                   PREC const &z) const
{
    return 1.0;
}

__device__
Direction DirectionalLight::dirFromDevice(PREC const &x, PREC const &y, PREC const &z, PREC &dist,
                                          curandState_t * const &state) const
{

    return dirFrom(x, y, z, dist);
}

Direction DirectionalLight::dirFromHost(PREC const &x, PREC const &y, PREC const &z, PREC &dist) const
{

    return dirFrom(x, y, z, dist);
}

PX_CUDA_CALLABLE
Direction DirectionalLight::dirFrom(PREC const &x,
                                    PREC const &y,
                                    PREC const &z,
                                    PREC &dist) const
{
    dist = FLT_MAX;
    return _neg_dir;
}

void DirectionalLight::up2Gpu()
{
#ifdef USE_CUDA
    if (_need_upload || BaseLight::need_upload)
    {
        if (dev_ptr == nullptr)
            PX_CUDA_CHECK(cudaMalloc(&dev_ptr, sizeof(BaseLight**)));

        GpuCreator::DirectionalLight(dev_ptr, _light, _dir);

        BaseLight::need_upload = false;
        _need_upload = false;
    }
#endif
}

void DirectionalLight::clearGpuData()
{
#ifdef USE_CUDA
    if (dev_ptr == nullptr)
        return;

    GpuCreator::destroy(dev_ptr);

    dev_ptr = nullptr;
    _need_upload = true;
#endif
}

void DirectionalLight::setDirection(Direction const &dir)
{
    _dir = dir;
    _neg_dir = dir * -1;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}

std::shared_ptr<BaseLight> PointLight::create(Light const &light,
                                              Point const &pos)
{
    return std::shared_ptr<BaseLight>(new PointLight(light, pos));
}

PointLight::PointLight(Light const &light, Point const &pos)
        : BaseLight(light),
          TYPE(Type::PointLight),
          _position(pos),
          _need_upload(true)

{}

PX_CUDA_CALLABLE
PREC PointLight::attenuate(PREC const &x, PREC const &y, PREC const &z) const
{
    auto nrm2 = (_position.x - x)*(_position.x - x) +
                (_position.y - y)*(_position.y - y) +
                (_position.z - z)*(_position.z - z);
    return nrm2 < EPSILON ? FLT_MAX : 1.0 / nrm2;
}

__device__
Direction PointLight::dirFromDevice(PREC const &x, PREC const &y, PREC const &z, PREC &dist,
                                    curandState_t * const &state) const
{

    return dirFrom(x, y, z, dist);
}

Direction PointLight::dirFromHost(PREC const &x, PREC const &y, PREC const &z, PREC &dist) const
{

    return dirFrom(x, y, z, dist);
}

PX_CUDA_CALLABLE
Direction PointLight::dirFrom(PREC const &x, PREC const &y, PREC const &z, PREC &dist) const
{
    auto dx = _position.x - x;
    auto dy = _position.y - y;
    auto dz = _position.z - z;

    dist = std::sqrt(dx*dx + dy*dy + dz*dz);
    return Direction(dx, dy, dz);
}

void PointLight::up2Gpu()
{
#ifdef USE_CUDA
    if (_need_upload || BaseLight::need_upload)
    {
        if (dev_ptr == nullptr)
            PX_CUDA_CHECK(cudaMalloc(&dev_ptr, sizeof(BaseLight**)));

        GpuCreator::PointLight(dev_ptr, _light, _position);
        
        BaseLight::need_upload = false;
        _need_upload = false;
    }
#endif
}

void PointLight::clearGpuData()
{
#ifdef USE_CUDA
    if (dev_ptr == nullptr)
        return;

    GpuCreator::destroy(dev_ptr);
    
    dev_ptr = nullptr;
    _need_upload = true;
#endif
}
void PointLight::setPosition(Point const &p)
{
    _position = p;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}

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
        : BaseLight(light),
          TYPE(Type::PointLight),
          _position(pos),
          _direction(direction),
          _falloff(falloff),
          _need_upload(true)
{
    setAngles(half_angle1, half_angle2);
}

PX_CUDA_CALLABLE
PREC SpotLight::attenuate(PREC const &x,
                            PREC const &y,
                            PREC const &z) const
{
    PREC dx = x-_position.x;
    PREC dy = y-_position.y;
    PREC dz = z-_position.z;
    PREC nrm2 = dx*dx + dy*dy + dz*dz;
    if (nrm2 < EPSILON)
        return FLT_MAX;

    PREC nrm = std::sqrt(nrm2);

    dx /= nrm;
    dy /= nrm;
    dz /= nrm;

    PREC cosine = _direction.x * dx + _direction.y * dy + _direction.z * dz;

    if (cosine >= _inner_ha_cosine)
        return 1.0/nrm2;
    if (cosine > _outer_ha_cosine)
        return std::pow(((_outer_ha_cosine-cosine)*_multiplier), _falloff)/nrm2;
    return 0;
}

__device__
Direction SpotLight::dirFromDevice(PREC const &x, PREC const &y, PREC const &z, PREC &dist,
                                   curandState_t * const &state) const
{

    return dirFrom(x, y, z, dist);
}

Direction SpotLight::dirFromHost(PREC const &x, PREC const &y, PREC const &z, PREC &dist) const
{

    return dirFrom(x, y, z, dist);
}

PX_CUDA_CALLABLE
Direction SpotLight::dirFrom(PREC const &x, PREC const &y, PREC const &z, PREC &dist) const
{
    auto dx = _position.x - x;
    auto dy = _position.y - y;
    auto dz = _position.z - z;

    dist = std::sqrt(dx*dx + dy*dy + dz*dz);
    return Direction(dx, dy, dz);
}

void SpotLight::up2Gpu()
{
#ifdef USE_CUDA
    if (_need_upload || BaseLight::need_upload)
    {
        if (dev_ptr == nullptr)
            PX_CUDA_CHECK(cudaMalloc(&dev_ptr, sizeof(BaseLight**)));

        GpuCreator::SpotLight(dev_ptr,
                              _light, _position, _direction,
                              _inner_ha, _outer_ha, _falloff);

        BaseLight::need_upload = false;
        _need_upload = false;
    }
#endif
}

void SpotLight::clearGpuData()
{
#ifdef USE_CUDA
    if (dev_ptr == nullptr)
        return;

    GpuCreator::destroy(dev_ptr);

    dev_ptr = nullptr;
    _need_upload = true;
#endif
}
void SpotLight::setPosition(Point const &pos)
{
    _position = pos;

#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void SpotLight::setDirection(Direction const &direction)
{
    _direction = direction;

#ifdef USE_CUDA
    _need_upload = true;
#endif
}

PX_CUDA_CALLABLE
void SpotLight::setAngles(PREC const &half_angle1, PREC const &half_angle2)
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

#ifdef USE_CUDA
    _need_upload = true;
#endif
}

void SpotLight::setFalloff(PREC const &falloff)
{
    _falloff = falloff;

#ifdef USE_CUDA
    _need_upload = true;
#endif
}
std::shared_ptr<BaseLight> AreaLight::create(Light const &light,
                                             Point const &pos,
                                             PREC const &radius)
{
    return std::shared_ptr<BaseLight>(new AreaLight(light, pos, radius));
}

AreaLight::AreaLight(Light const &light,
                         Point const &center,
                         PREC const &radius)
        : BaseLight(light),
          TYPE(Type::AreaLight),
          _center(center),
          _radius(radius),
          _need_upload(true)

{}

PX_CUDA_CALLABLE
PREC AreaLight::attenuate(PREC const &x, PREC const &y, PREC const &z) const
{
    auto nrm2 = (_center.x - x)*(_center.x - x) +
                (_center.y - y)*(_center.y - y) +
                (_center.z - z)*(_center.z - z);
    return nrm2 < EPSILON ? FLT_MAX : 1.0 / 4 / PI / std::sqrt(nrm2);
}

__device__
Direction AreaLight::dirFromDevice(PREC const &x, PREC const &y, PREC const &z, PREC &dist,
                                   curandState_t * const &state) const
{

    auto dx = _center.x - x + _radius * rnd::rnd_gpu(state);
    auto dy = _center.y - y + _radius * rnd::rnd_gpu(state);
    auto dz = _center.z - z + _radius * rnd::rnd_gpu(state);

    dist = std::sqrt(dx * dx + dy * dy + dz * dz);
    return Direction(dx, dy, dz);
}

Direction AreaLight::dirFromHost(PREC const &x, PREC const &y, PREC const &z, PREC &dist) const
{
    auto dx = _center.x - x + _radius * rnd::rnd_cpu();
    auto dy = _center.y - y + _radius * rnd::rnd_cpu();
    auto dz = _center.z - z + _radius * rnd::rnd_cpu();

    dist = std::sqrt(dx*dx + dy*dy + dz*dz);
    return Direction(dx, dy, dz);
}

void AreaLight::up2Gpu()
{
#ifdef USE_CUDA
    if (_need_upload || BaseLight::need_upload)
    {
        if (dev_ptr == nullptr)
            PX_CUDA_CHECK(cudaMalloc(&dev_ptr, sizeof(BaseLight**)));

        GpuCreator::AreaLight(dev_ptr, _light, _center, _radius);

        BaseLight::need_upload = false;
        _need_upload = false;
    }
#endif
}

void AreaLight::clearGpuData()
{
#ifdef USE_CUDA
    if (dev_ptr == nullptr)
        return;

    GpuCreator::destroy(dev_ptr);

    dev_ptr = nullptr;
    _need_upload = true;
#endif
}
void AreaLight::setCenter(Point const &p)
{
    _center = p;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}

void AreaLight::setRadius(const PREC &radius)
{
    _radius = radius;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}