#include "object/light/directional_light.hpp"

#include <cfloat>

using namespace px;

BaseDirectionalLight::BaseDirectionalLight(Direction const &dir)
    : _dir(dir), _neg_dir(dir * -1)
{}

void BaseDirectionalLight::setDir(Direction const &dir)
{
    _dir = dir;
    _neg_dir = dir * -1;
}

PX_CUDA_CALLABLE
PREC BaseDirectionalLight::attenuate(void *const &obj,
                                     PREC const &x, PREC const &y, PREC const &z)
{
    return 1.0;
}

Direction BaseDirectionalLight::dirFromHost(BaseDirectionalLight *const &obj,
                                            PREC const &x, PREC const &y,
                                            PREC const &z, PREC &dist)
{
    dist = FLT_MAX;
    return obj->_neg_dir;
}

PX_CUDA_CALLABLE
Direction BaseDirectionalLight::dirFromDevice(void *const &obj, PREC const &x,
                                              PREC const &y, PREC const &z,
                                              PREC &dist,
                                              curandState_t *const &)
{
    dist = FLT_MAX;
    return reinterpret_cast<BaseDirectionalLight*>(obj)->_neg_dir;
}

const LightType DirectionalLight::TYPE = LightType::DirectionalLight;

std::shared_ptr<BaseLight> DirectionalLight::create(Light const &light,
                                                    Direction const &dir)
{
    return std::shared_ptr<BaseLight>(new DirectionalLight(light, dir));
}

DirectionalLight::DirectionalLight(Light const &light, Direction const &dir)
        : BaseLight(TYPE, light),
          _obj(new BaseDirectionalLight(dir)),
          _gpu_obj(nullptr),
          _need_upload(true)
{}

DirectionalLight::~DirectionalLight()
{
    delete _obj;
#ifdef USE_CUDA
    clearGpuData();
#endif
}

PREC DirectionalLight::attenuate(PREC const &x, PREC const &y, PREC const &z) const
{
    return BaseDirectionalLight::attenuate(_obj, x, y, z);
}

Direction DirectionalLight::dirFrom(PREC const &x, PREC const &y, PREC const &z, PREC &dist) const
{
    return BaseDirectionalLight::dirFromHost(_obj, x, y, z, dist);
}

void DirectionalLight::setDir(Direction const &dir)
{
    _obj->setDir(dir);
#ifdef USE_CUDA
    _need_upload = true;
#endif
}

#ifdef USE_CUDA
__device__ fnAttenuate_t __fn_attenuate_directional_light = BaseDirectionalLight::attenuate;
__device__ fnDirFrom_t  __fn_dir_from_directional_light = BaseDirectionalLight::dirFromDevice;
#endif

void DirectionalLight::up2Gpu()
{
#ifdef USE_CUDA
    static fnAttenuate_t fn_attenuate_h = nullptr;
    static fnDirFrom_t  fn_dir_from_h;
    if (_need_upload || BaseLight::need_upload)
    {
        if (dev_ptr == nullptr)
        {
            PX_CUDA_CHECK(cudaMalloc(&_gpu_obj, sizeof(BaseDirectionalLight)));
            PX_CUDA_CHECK(cudaMalloc(&dev_ptr, sizeof(LightObj)));
        }
        if (fn_attenuate_h == nullptr)
        {
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_attenuate_h, __fn_attenuate_directional_light, sizeof(fnAttenuate_t)));
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_dir_from_h, __fn_dir_from_directional_light, sizeof(fnDirFrom_t)));
        }
        PX_CUDA_CHECK(cudaMemcpy(_gpu_obj, _obj, sizeof(BaseDirectionalLight),
                                 cudaMemcpyHostToDevice));

        LightObj tmp(_gpu_obj, type, _light, fn_attenuate_h, fn_dir_from_h);

        PX_CUDA_CHECK(cudaMemcpy(dev_ptr, &tmp, sizeof(LightObj),
                                 cudaMemcpyHostToDevice));

        _need_upload = false;
        BaseLight::need_upload = false;
    }
#endif
}

void DirectionalLight::clearGpuData()
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
