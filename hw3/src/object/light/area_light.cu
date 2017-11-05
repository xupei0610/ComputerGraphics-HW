#include "object/light/area_light.hpp"

#include <cfloat>

using namespace px;

BaseAreaLight::BaseAreaLight(Point const &center,
                             PREC const &radius)
        : _center(center), _radius(radius), _radius2(radius*radius)
{}


void BaseAreaLight::setCenter(Point const &center)
{
    _center = center;
}

void BaseAreaLight::setRadius(PREC const &r)
{
    _radius = r;
    _radius2 = r*r;
}

PX_CUDA_CALLABLE
PREC BaseAreaLight::attenuate(void *const &obj,
                              PREC const &x, PREC const &y, PREC const &z)
{
    auto o = reinterpret_cast<BaseAreaLight*>(obj);
    auto nrm2 = (o->_center.x - x)*(o->_center.x - x) +
                (o->_center.y - y)*(o->_center.y - y) +
                (o->_center.z - z)*(o->_center.z - z);
    return nrm2 < o->_radius2 ? FLT_MAX : PREC(0.25) / (PREC(PI) * std::sqrt(nrm2));
}

Direction BaseAreaLight::dirFromHost(BaseAreaLight *const &obj,
                                     PREC const &x, PREC const &y,
                                     PREC const &z, PREC &dist)
{
    auto dx = obj->_center.x - x;
    auto dy = obj->_center.y - y;
    auto dz = obj->_center.z - z;

    dist = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (dist < obj->_radius)
    {
        // inside the light source
        dist = 0;
        return {0, 0, 0};
    }
    auto r = obj->_radius * rnd::rnd_cpu();
    dx += r;
    dist += r*r;
    r = obj->_radius * rnd::rnd_cpu();
    dy += r;
    dist += r*r;
    r = obj->_radius * rnd::rnd_cpu();
    dz += r;
    dist += r*r;
    return {dx, dy, dz};
}

__device__
Direction BaseAreaLight::dirFromDevice(void *const &obj, PREC const &x,
                                       PREC const &y, PREC const &z,
                                       PREC &dist,
                                       curandState_t * const &state)
{
    auto o = reinterpret_cast<BaseAreaLight*>(obj);

    auto dx = o->_center.x - x;// + _radius * rnd::rnd_gpu(state);
    auto dy = o->_center.y - y;// + _radius * rnd::rnd_gpu(state);
    auto dz = o->_center.z - z;// + _radius * rnd::rnd_gpu(state);

    dist = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (dist < o->_radius)
    {
        // inside the light source
        dist = 0;
        return {0, 0, 0};
    }
    auto r = o->_radius * rnd::rnd_gpu(state);
    dx += r;
    dist += r*r;
    r = o->_radius * rnd::rnd_gpu(state);
    dy += r;
    dist += r*r;
    r = o->_radius * rnd::rnd_gpu(state);
    dz += r;
    dist += r*r;
    return {dx, dy, dz};
}

#ifdef USE_CUDA
__device__ fnAttenuate_t __fn_attenuate_area_light = BaseAreaLight::attenuate;
__device__ fnDirFrom_t  __fn_dir_from_area_light = BaseAreaLight::dirFromDevice;
#endif

const LightType AreaLight::TYPE = LightType::AreaLight;

std::shared_ptr<BaseLight> AreaLight::create(Light const &light,
                                             Point const &center,
                                             PREC const &radius)
{
    return std::shared_ptr<BaseLight>(new AreaLight(light, center, radius));
}

AreaLight::AreaLight(Light const &light,
                     Point const &center,
                     PREC const &radius)
        : BaseLight(TYPE, light),
          _obj(new BaseAreaLight(center, radius)),
          _gpu_obj(nullptr),
          _need_upload(true)
{}

AreaLight::~AreaLight()
{
    delete _obj;
#ifdef USE_CUDA
    clearGpuData();
#endif
}

PREC AreaLight::attenuate(PREC const &x, PREC const &y, PREC const &z) const
{
    return BaseAreaLight::attenuate(_obj, x, y, z);
}

Direction AreaLight::dirFrom(PREC const &x, PREC const &y, PREC const &z, PREC &dist) const
{
    return BaseAreaLight::dirFromHost(_obj, x, y, z, dist);
}

void AreaLight::setCenter(Point const &c)
{
    _obj->setCenter(c);
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void AreaLight::setRadius(PREC const &r)
{
    _obj->setRadius(r);
#ifdef USE_CUDA
    _need_upload = true;
#endif
}

void AreaLight::up2Gpu()
{
#ifdef USE_CUDA
    static fnAttenuate_t fn_attenuate_h = nullptr;
    static fnDirFrom_t  fn_dir_from_h;
    if (_need_upload || BaseLight::need_upload)
    {
        if (dev_ptr == nullptr)
        {
            PX_CUDA_CHECK(cudaMalloc(&_gpu_obj, sizeof(BaseAreaLight)));
            PX_CUDA_CHECK(cudaMalloc(&dev_ptr, sizeof(LightObj)));
        }
        if (fn_attenuate_h == nullptr)
        {
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_attenuate_h, __fn_attenuate_area_light, sizeof(fnAttenuate_t)));
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_dir_from_h, __fn_dir_from_area_light, sizeof(fnDirFrom_t)));
        }
        PX_CUDA_CHECK(cudaMemcpy(_gpu_obj, _obj, sizeof(BaseAreaLight),
                                 cudaMemcpyHostToDevice));

        LightObj tmp(_gpu_obj, type, _light, fn_attenuate_h, fn_dir_from_h);

        PX_CUDA_CHECK(cudaMemcpy(dev_ptr, &tmp, sizeof(LightObj),
                                 cudaMemcpyHostToDevice));

        _need_upload = false;
        BaseLight::need_upload = false;
    }
#endif
}

void AreaLight::clearGpuData()
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
