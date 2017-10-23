#include <cfloat>
#include "object/light/light.hpp"

using namespace px;

BaseLight::BaseLight(Light const &light)
        : _light(light), need_upload(true)
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
          _dev_ptr(nullptr),
          _need_upload(true)
{}

DirectionalLight::~DirectionalLight()
{
    clearGpuData();
}

PX_CUDA_CALLABLE
double DirectionalLight::attenuate(double const &x,
                                   double const &y,
                                   double const &z) const
{
    return 1.0;
}

__device__
Direction DirectionalLight::dirFromDevice(double const &x, double const &y, double const &z, double &dist) const
{

    return dirFrom(x, y, z, dist);
}

Direction DirectionalLight::dirFromHost(double const &x, double const &y, double const &z, double &dist) const
{

    return dirFrom(x, y, z, dist);
}

PX_CUDA_CALLABLE
Direction DirectionalLight::dirFrom(double const &x,
                                    double const &y,
                                    double const &z,
                                    double &dist) const
{
    dist = FLT_MAX;
    return _neg_dir;
}

BaseLight* DirectionalLight::up2Gpu()
{
#ifdef USE_CUDA
    if (_need_upload || BaseLight::need_upload)
    {
        if (_dev_ptr == nullptr)
            PX_CUDA_CHECK(cudaMalloc(&_dev_ptr, sizeof(DirectionalLight)));

        PX_CUDA_CHECK(cudaMemcpy(_dev_ptr, this, sizeof(DirectionalLight),
                                 cudaMemcpyHostToDevice));
        BaseLight::need_upload = false;
        _need_upload = false;
    }
    return _dev_ptr;
#else
    return this;
#endif
}

void DirectionalLight::clearGpuData()
{
#ifdef USE_CUDA
    if (_dev_ptr == nullptr)
        return;

    PX_CUDA_CHECK(cudaFree(_dev_ptr));
    _dev_ptr = nullptr;
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

PointLight::~PointLight()
{
    clearGpuData();
}

PX_CUDA_CALLABLE
double PointLight::attenuate(double const &x, double const &y, double const &z) const
{
    auto nrm2 = (_position.x - x)*(_position.x - x) +
                (_position.y - y)*(_position.y - y) +
                (_position.z - z)*(_position.z - z);
    return nrm2 < FLT_MIN ? FLT_MAX : 1.0 / nrm2;
}

__device__
Direction PointLight::dirFromDevice(double const &x, double const &y, double const &z, double &dist) const
{

    return dirFrom(x, y, z, dist);
}

Direction PointLight::dirFromHost(double const &x, double const &y, double const &z, double &dist) const
{

    return dirFrom(x, y, z, dist);
}

PX_CUDA_CALLABLE
Direction PointLight::dirFrom(double const &x, double const &y, double const &z, double &dist) const
{
    auto dx = _position.x - x;
    auto dy = _position.y - y;
    auto dz = _position.z - z;

    dist = std::sqrt(dx*dx + dy*dy + dz*dz);
    return Direction(dx, dy, dz);
}

BaseLight* PointLight::up2Gpu()
{
#ifdef USE_CUDA
    if (_need_upload || BaseLight::need_upload)
    {
        if (_dev_ptr == nullptr)
            PX_CUDA_CHECK(cudaMalloc(&_dev_ptr, sizeof(PointLight)));

        PX_CUDA_CHECK(cudaMemcpy(_dev_ptr, this, sizeof(PointLight),
                                 cudaMemcpyHostToDevice));
        BaseLight::need_upload = false;
        _need_upload = false;
    }
    return _dev_ptr;
#else
    return this;
#endif
}

void PointLight::clearGpuData()
{
#ifdef USE_CUDA
    if (_dev_ptr == nullptr)
        return;

    PX_CUDA_CHECK(cudaFree(_dev_ptr));
    _dev_ptr = nullptr;
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
                                             double const &half_angle1,
                                             double const &half_angle2,
                                             double const &falloff)
{
    return std::shared_ptr<BaseLight>(new SpotLight(light, pos, direction,
                                                    half_angle1, half_angle2,
                                                    falloff));
}

SpotLight::SpotLight(Light const &light,
                     Point const &pos,
                     Direction const &direction,
                     double const &half_angle1,
                     double const &half_angle2,
                     double const &falloff)
        : BaseLight(light),
          TYPE(Type::PointLight),
          _position(pos),
          _direction(direction),
          _falloff(falloff),
          _need_upload(true)
{
    setAngles(half_angle1, half_angle2);
}

SpotLight::~SpotLight()
{
    clearGpuData();
}

PX_CUDA_CALLABLE
double SpotLight::attenuate(double const &x,
                            double const &y,
                            double const &z) const
{
    double dx = x-_position.x;
    double dy = y-_position.y;
    double dz = z-_position.z;
    double nrm2 = dx*dx + dy*dy + dz*dz;
    if (nrm2 < FLT_MIN)
        return FLT_MAX;

    double nrm = std::sqrt(nrm2);

    dx /= nrm;
    dy /= nrm;
    dz /= nrm;

    double cosine = _direction.x * dx + _direction.y * dy + _direction.z * dz;

    if (cosine >= _inner_ha_cosine)
        return 1.0/nrm2;
    if (cosine > _outer_ha_cosine)
        return std::pow(((_outer_ha_cosine-cosine)*_multiplier), _falloff)/nrm2;
    return 0;
}

__device__
Direction SpotLight::dirFromDevice(double const &x, double const &y, double const &z, double &dist) const
{

    return dirFrom(x, y, z, dist);
}

Direction SpotLight::dirFromHost(double const &x, double const &y, double const &z, double &dist) const
{

    return dirFrom(x, y, z, dist);
}

PX_CUDA_CALLABLE
Direction SpotLight::dirFrom(double const &x, double const &y, double const &z, double &dist) const
{
    auto dx = _position.x - x;
    auto dy = _position.y - y;
    auto dz = _position.z - z;

    dist = std::sqrt(dx*dx + dy*dy + dz*dz);
    return Direction(dx, dy, dz);
}

BaseLight* SpotLight::up2Gpu()
{
#ifdef USE_CUDA
    if (_need_upload || BaseLight::need_upload)
    {
        if (_dev_ptr == nullptr)
            PX_CUDA_CHECK(cudaMalloc(&_dev_ptr, sizeof(SpotLight)));

        PX_CUDA_CHECK(cudaMemcpy(_dev_ptr, this, sizeof(SpotLight),
                                 cudaMemcpyHostToDevice));
        BaseLight::need_upload = false;
        _need_upload = false;
    }
    return _dev_ptr;
#else
    return this;
#endif
}

void SpotLight::clearGpuData()
{
#ifdef USE_CUDA
    if (_dev_ptr == nullptr)
        return;

    PX_CUDA_CHECK(cudaFree(_dev_ptr));
    _dev_ptr = nullptr;
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

void SpotLight::setAngles(double const &half_angle1, double const &half_angle2)
{
    _inner_ha = half_angle1 < 0 ?
                std::fmod(half_angle1, PI2) + PI2 : std::fmod(half_angle1, PI2);
    _outer_ha = half_angle2 < 0 ?
                std::fmod(half_angle2, PI2) + PI2 : std::fmod(half_angle2, PI2);

    if (_inner_ha_cosine > PI)
        _inner_ha_cosine = PI;
    if (_outer_ha_cosine > PI)
        _outer_ha_cosine = PI;

    if (_outer_ha < _inner_ha)
        std::swap(_outer_ha, _inner_ha);

    _inner_ha_cosine = std::cos(_inner_ha);
    _outer_ha_cosine = std::cos(_outer_ha);
    _multiplier = 1.0 / (_outer_ha_cosine - _inner_ha_cosine);

#ifdef USE_CUDA
    _need_upload = true;
#endif
}

void SpotLight::setFalloff(double const &falloff)
{
    _falloff = falloff;

#ifdef USE_CUDA
    _need_upload = true;
#endif
}

std::shared_ptr<BaseLight> AreaLight::create(Light const &light,
                                             Point const &center,
                                             double const &radius)
{
    return std::shared_ptr<BaseLight>(new AreaLight(light, center, radius));
}

AreaLight::AreaLight(Light const &light,
                     Point const &center,
                     double const &radius)
        : BaseLight(light),
          TYPE(Type::AreaLight),
          _center(center),
          _radius(radius),
          _need_upload(true)
{}

AreaLight::~AreaLight()
{
    clearGpuData();
}

PX_CUDA_CALLABLE
double AreaLight::attenuate(double const &x, double const &y, double const &z) const
{
    auto nrm2 = (_center.x - x)*(_center.x - x) +
                (_center.y - y)*(_center.y - y) +
                (_center.z - z)*(_center.z - z);
    return nrm2 < FLT_MIN ? FLT_MAX : 1.0 / nrm2;
}

Direction AreaLight::dirFromHost(double const &x, double const &y, double const &z, double &dist) const
{
    auto dx = _center.x + _radius * RND::rnd_cpu();
    auto dy = _center.y + _radius * RND::rnd_cpu();
    auto dz = _center.z + _radius * RND::rnd_cpu();

    dx -= x;
    dy -= y;
    dz -= z;

    dist = std::sqrt(dx*dx + dy*dy + dz*dz);
    return Direction(dx, dy, dz);
}

__device__
Direction AreaLight::dirFromDevice(double const &x, double const &y, double const &z, double &dist) const
{
    auto dx = _center.x + _radius * RND::rnd_gpu();
    auto dy = _center.y + _radius * RND::rnd_gpu();
    auto dz = _center.z + _radius * RND::rnd_gpu();

    dx -= x;
    dy -= y;
    dz -= z;

    dist = std::sqrt(dx*dx + dy*dy + dz*dz);
    return Direction(dx, dy, dz);
}

BaseLight* AreaLight::up2Gpu()
{
#ifdef USE_CUDA
    if (_need_upload || BaseLight::need_upload)
    {
        if (_dev_ptr == nullptr)
            PX_CUDA_CHECK(cudaMalloc(&_dev_ptr, sizeof(AreaLight)));

        PX_CUDA_CHECK(cudaMemcpy(_dev_ptr, this, sizeof(AreaLight),
                                 cudaMemcpyHostToDevice));
        BaseLight::need_upload = false;
        _need_upload = false;
    }
    return _dev_ptr;
#else
    return this;
#endif
}

void AreaLight::clearGpuData()
{
#ifdef USE_CUDA
    if (_dev_ptr == nullptr)
        return;

    PX_CUDA_CHECK(cudaFree(_dev_ptr));
    _dev_ptr = nullptr;
    _need_upload = true;
#endif
}

void AreaLight::setCenter(Point const &center)
{
    _center = center;

#ifdef USE_CUDA
    _need_upload = true;
#endif
}

void AreaLight::setRadius(double const &radius)
{
    _radius = radius;

#ifdef USE_CUDA
    _need_upload = true;
#endif
}