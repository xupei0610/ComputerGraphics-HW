#include "object/light/point_light.hpp"

#include <cfloat>

using namespace px;

BasePointLight::BasePointLight(Point const &pos)
        : _pos(pos)
{}

void BasePointLight::setPos(Point const &pos)
{
    _pos = pos;
}

PX_CUDA_CALLABLE
PREC BasePointLight::attenuate(void *const &obj,
                                     PREC const &x, PREC const &y, PREC const &z)
{
    auto o = reinterpret_cast<BasePointLight*>(obj);
    auto nrm2 = (o->_pos.x - x)*(o->_pos.x - x) +
                (o->_pos.y - y)*(o->_pos.y - y) +
                (o->_pos.z - z)*(o->_pos.z - z);
    return nrm2 < EPSILON ? FLT_MAX : 1.0 / nrm2;
}

Direction BasePointLight::dirFromHost(BasePointLight *const &obj,
                                            PREC const &x, PREC const &y,
                                            PREC const &z, PREC &dist)
{
    auto dx = obj->_pos.x - x;
    auto dy = obj->_pos.y - y;
    auto dz = obj->_pos.z - z;

    dist = std::sqrt(dx*dx + dy*dy + dz*dz);
    return Direction(dx, dy, dz);
}

PX_CUDA_CALLABLE
Direction BasePointLight::dirFromDevice(void *const &obj,
                                        PREC const &x, PREC const &y, PREC const &z,
                                        PREC &dist,
                                        curandState_t * const &)
{
    auto o = reinterpret_cast<BasePointLight*>(obj);
    auto dx = o->_pos.x - x;
    auto dy = o->_pos.y - y;
    auto dz = o->_pos.z - z;

    dist = std::sqrt(dx*dx + dy*dy + dz*dz);
    return Direction(dx, dy, dz);
}

const LightType PointLight::TYPE = LightType::PointLight;

std::shared_ptr<BaseLight> PointLight::create(Light const &light,
                                              Point const &pos)
{
    return std::shared_ptr<BaseLight>(new PointLight(light, pos));
}

PointLight::PointLight(Light const &light, Point const &pos)
        : BaseLight(TYPE, light),
          _obj(new BasePointLight(pos)),
          _gpu_obj(nullptr),
          _need_upload(true)
{}

PointLight::~PointLight()
{
    delete _obj;
#ifdef USE_CUDA
    clearGpuData();
#endif
}

PREC PointLight::attenuate(PREC const &x, PREC const &y, PREC const &z) const
{
    return BasePointLight::attenuate(_obj, x, y, z);
}

Direction PointLight::dirFrom(PREC const &x, PREC const &y, PREC const &z, PREC &dist) const
{
    return BasePointLight::dirFromHost(_obj, x, y, z, dist);
}

void PointLight::setPos(Point const &pos)
{
    _obj->setPos(pos);
#ifdef USE_CUDA
    _need_upload = true;
#endif
}

#ifdef USE_CUDA
__device__ fnAttenuate_t __fn_attenuate_point_light = BasePointLight::attenuate;
__device__ fnDirFrom_t  __fn_dir_from_point_light = BasePointLight::dirFromDevice;
#endif

void PointLight::up2Gpu()
{
#ifdef USE_CUDA
    static fnAttenuate_t fn_attenuate_h = nullptr;
    static fnDirFrom_t  fn_dir_from_h;
    if (_need_upload || BaseLight::need_upload)
    {
        if (dev_ptr == nullptr)
        {
            PX_CUDA_CHECK(cudaMalloc(&_gpu_obj, sizeof(BasePointLight)));
            PX_CUDA_CHECK(cudaMalloc(&dev_ptr, sizeof(LightObj)));
        }
        if (fn_attenuate_h == nullptr)
        {
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_attenuate_h, __fn_attenuate_point_light, sizeof(fnAttenuate_t)));
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_dir_from_h, __fn_dir_from_point_light, sizeof(fnDirFrom_t)));
        }
        PX_CUDA_CHECK(cudaMemcpy(_gpu_obj, _obj, sizeof(BasePointLight),
                                 cudaMemcpyHostToDevice));

        LightObj tmp(_gpu_obj, type, _light, fn_attenuate_h, fn_dir_from_h);

        PX_CUDA_CHECK(cudaMemcpy(dev_ptr, &tmp, sizeof(LightObj),
                                 cudaMemcpyHostToDevice));

        _need_upload = false;
        BaseLight::need_upload = false;
    }
#endif
}

void PointLight::clearGpuData()
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