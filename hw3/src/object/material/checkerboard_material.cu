#include "object/material/checkerboard_material.hpp"

using namespace px;

BaseCheckerboardMaterial::BaseCheckerboardMaterial(Light const &ambient,
                                                   Light const &diffuse,
                                                   Light const &specular,
                                                   PREC const &shininessonent,
                                                   Light const &transmissive,
                                                   PREC const &refractive_index,
                                                   PREC const &dim_scale,
                                                   PREC const &color_scale)
        : _ambient(ambient),
          _diffuse(diffuse),
          _specular(specular),
          _shininessonent(shininessonent),
          _transmissive(transmissive),
          _refractive_index(refractive_index),
          _dim_scale(dim_scale),
          _color_scale(color_scale)
{}

PX_CUDA_CALLABLE
Light BaseCheckerboardMaterial::getAmbient(void *const &obj, PREC const &u, PREC const &v, PREC const &w)
{
    auto o = reinterpret_cast<BaseCheckerboardMaterial*>(obj);
    if (((u > 0 ? std::fmod(u * o->_dim_scale, static_cast<decltype(o->_dim_scale)>(1.0)) > 0.5 :
          std::fmod(-u * o->_dim_scale, static_cast<decltype(o->_dim_scale)>(1.0)) <= 0.5) ^
         (v > 0 ? std::fmod(v * o->_dim_scale, static_cast<decltype(o->_dim_scale)>(1.0)) > 0.5 :
          std::fmod(-v * o->_dim_scale, static_cast<decltype(o->_dim_scale)>(1.0)) <= 0.5)) == 1)
        return o->_ambient;
    return o->_ambient*o->_color_scale;
}

PX_CUDA_CALLABLE
Light BaseCheckerboardMaterial::getDiffuse(void *const &obj, PREC const &u, PREC const &v, PREC const &w)
{
    auto o = reinterpret_cast<BaseCheckerboardMaterial*>(obj);
    if (((u > 0 ? std::fmod(u * o->_dim_scale, static_cast<decltype(o->_dim_scale)>(1.0)) > 0.5 :
          std::fmod(-u * o->_dim_scale, static_cast<decltype(o->_dim_scale)>(1.0)) <= 0.5) ^
         (v > 0 ? std::fmod(v * o->_dim_scale, static_cast<decltype(o->_dim_scale)>(1.0)) > 0.5 :
          std::fmod(-v * o->_dim_scale, static_cast<decltype(o->_dim_scale)>(1.0)) <= 0.5)) == 1)
        return o->_diffuse;
    return o->_diffuse*o->_color_scale;
}

PX_CUDA_CALLABLE
Light BaseCheckerboardMaterial::getSpecular(void *const &obj, PREC const &u, PREC const &v, PREC const &w)
{
    return reinterpret_cast<BaseCheckerboardMaterial*>(obj)->_specular;
}

PX_CUDA_CALLABLE
PREC BaseCheckerboardMaterial::getShininess(void *const &obj, PREC const &u, PREC const &v, PREC const &w)
{
    return reinterpret_cast<BaseCheckerboardMaterial*>(obj)->_shininessonent;
}

PX_CUDA_CALLABLE
Light BaseCheckerboardMaterial::getTransmissive(void *const &obj, PREC const &u, PREC const &v, PREC const &w)
{
    return reinterpret_cast<BaseCheckerboardMaterial*>(obj)->_transmissive;
}

PX_CUDA_CALLABLE
PREC BaseCheckerboardMaterial::getRefractiveIndex(void *const &obj, PREC const &u, PREC const &v, PREC const &w)
{
    return reinterpret_cast<BaseCheckerboardMaterial*>(obj)->_refractive_index;
}

void BaseCheckerboardMaterial::setAmbient(Light const &ambient)
{
    _ambient = ambient;
}
void BaseCheckerboardMaterial::setDiffuse(Light const &diffuse)
{
    _diffuse = diffuse;
}
void BaseCheckerboardMaterial::setSpecular(Light const &specular)
{
    _specular = specular;
}
void BaseCheckerboardMaterial::setShininess(PREC const &shininess)
{
    _shininessonent = shininess;
}
void BaseCheckerboardMaterial::setTransmissive(Light const &transmissive)
{
    _transmissive = transmissive;
}
void BaseCheckerboardMaterial::setRefractiveIndex(PREC const &ior)
{
    _refractive_index = ior;
}
void BaseCheckerboardMaterial::setDimScale(PREC const &scale)
{
    _dim_scale = scale;
}
void BaseCheckerboardMaterial::setColorScale(PREC const &scale)
{
    _color_scale = scale;
}

std::shared_ptr<BaseMaterial> CheckerboardMaterial::create(Light const &ambient,
                                                       Light const &diffuse,
                                                       Light const &specular,
                                                       PREC const &shininessonent,
                                                       Light const &transmissive,
                                                       PREC const &refractive_index,
                                                       PREC const &dim_scale,
                                                       PREC const &color_scale)
{
    return std::shared_ptr<BaseMaterial>(new CheckerboardMaterial(ambient,
                                                       diffuse,
                                                       specular,
                                                       shininessonent,
                                                       transmissive,
                                                       refractive_index,
                                                             dim_scale, color_scale));
}

CheckerboardMaterial::CheckerboardMaterial(Light const &ambient,
                                           Light const &diffuse,
                                           Light const &specular,
                                           PREC const &shininessonent,
                                           Light const &transmissive,
                                           PREC const &refractive_index,
                                           PREC const &dim_scale,
                                           PREC const &color_scale)
        : BaseMaterial(),
          _obj(new BaseCheckerboardMaterial(ambient, diffuse,
                                     specular, shininessonent,
                                     transmissive, refractive_index,
                                            dim_scale, color_scale)),
          _gpu_obj(nullptr),
          _need_upload(true)
{}

CheckerboardMaterial::~CheckerboardMaterial()
{
    delete _obj;
#ifdef USE_CUDA
    clearGpuData();
#endif
}

#ifdef USE_CUDA
__device__ fnAmbient_t __fn_ambient_checkerboard_material = BaseCheckerboardMaterial::getAmbient;
__device__ fnDiffuse_t __fn_diffuse_checkerboard_material = BaseCheckerboardMaterial::getDiffuse;
__device__ fnSpecular_t __fn_specular_checkerboard_material = BaseCheckerboardMaterial::getSpecular;
__device__ fnShininess_t __fn_shininess_checkerboard_material = BaseCheckerboardMaterial::getShininess;
__device__ fnTransmissive_t __fn_transmissive_checkerboard_material = BaseCheckerboardMaterial::getTransmissive;
__device__ fnRefractiveIndex_t __fn_refractive_index_checkerboard_material = BaseCheckerboardMaterial::getRefractiveIndex;
#endif

void CheckerboardMaterial::up2Gpu()
{
#ifdef USE_CUDA
    static fnAmbient_t fn_ambient_h = nullptr;
    static fnDiffuse_t fn_diffuse_h;
    static fnSpecular_t fn_specular_h;
    static fnShininess_t fn_shininess_h;
    static fnTransmissive_t fn_transmissive_h;
    static fnRefractiveIndex_t fn_refractive_index_h;

    if (_need_upload)
    {
        if (dev_ptr == nullptr)
        {
            PX_CUDA_CHECK(cudaMalloc(&_gpu_obj, sizeof(BaseCheckerboardMaterial)));
            PX_CUDA_CHECK(cudaMalloc(&dev_ptr, sizeof(MaterialObj)));
        }
        if (fn_ambient_h == nullptr)
        {
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_ambient_h, __fn_ambient_checkerboard_material, sizeof(fnAmbient_t)));
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_diffuse_h, __fn_diffuse_checkerboard_material, sizeof(fnDiffuse_t)));
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_specular_h, __fn_specular_checkerboard_material, sizeof(fnSpecular_t)));
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_shininess_h, __fn_shininess_checkerboard_material, sizeof(fnShininess_t)));
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_transmissive_h, __fn_transmissive_checkerboard_material, sizeof(fnTransmissive_t)));
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_refractive_index_h, __fn_refractive_index_checkerboard_material, sizeof(fnRefractiveIndex_t)));
        }
        PX_CUDA_CHECK(cudaMemcpy(_gpu_obj, _obj, sizeof(BaseCheckerboardMaterial),
                                 cudaMemcpyHostToDevice));
        MaterialObj tmp(_gpu_obj,
                        fn_ambient_h, fn_diffuse_h,
                        fn_specular_h, fn_shininess_h,
                        fn_transmissive_h, fn_refractive_index_h);

        PX_CUDA_CHECK(cudaMemcpy(dev_ptr, &tmp, sizeof(MaterialObj),
                                 cudaMemcpyHostToDevice));
        _need_upload = false;
    }
#endif
}

void CheckerboardMaterial::clearGpuData()
{
#ifdef USE_CUDA
    if (_gpu_obj != nullptr)
    {
        _gpu_obj = nullptr;
        PX_CUDA_CHECK(cudaFree(_gpu_obj));
    }
    BaseMaterial::clearGpuData();
#endif
}

PREC CheckerboardMaterial::Shininess(PREC const &u, PREC const &v,
                                        PREC const &w) const
{
    return BaseCheckerboardMaterial::getShininess(_obj, u, v, w);
}
PREC CheckerboardMaterial::refractiveIndex(PREC const &u, PREC const &v, PREC const &w) const
{
    return BaseCheckerboardMaterial::getRefractiveIndex(_obj, u, v, w);
}
Light CheckerboardMaterial::getAmbient(PREC const &u, PREC const &v, PREC const &w) const
{
    return BaseCheckerboardMaterial::getAmbient(_obj, u, v, w);
}
Light CheckerboardMaterial::getDiffuse(PREC const &u, PREC const &v, PREC const &w) const
{
    return BaseCheckerboardMaterial::getDiffuse(_obj, u, v, w);
}
Light CheckerboardMaterial::getSpecular(PREC const &u, PREC const &v, PREC const &w) const
{
    return BaseCheckerboardMaterial::getSpecular(_obj, u, v, w);
}
Light CheckerboardMaterial::getTransmissive(PREC const &u, PREC const &v, PREC const &w) const
{
    return BaseCheckerboardMaterial::getTransmissive(_obj, u, v, w);
}

void CheckerboardMaterial::setAmbient(Light const &ambient)
{
    _obj->setAmbient(ambient);
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void CheckerboardMaterial::setDiffuse(Light const &diffuse)
{
    _obj->setDiffuse(diffuse);
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void CheckerboardMaterial::setSpecular(Light const &specular)
{
    _obj->setSpecular(specular);
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void CheckerboardMaterial::setShininess(PREC const &shininess)
{
    _obj->setShininess(shininess);
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void CheckerboardMaterial::setTransmissive(Light const &transmissive)
{
    _obj->setTransmissive(transmissive);
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void CheckerboardMaterial::setRefractiveIndex(PREC const &ior)
{
    _obj->setRefractiveIndex(ior);
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void CheckerboardMaterial::setDimScale(PREC const &scale)
{
    _obj->setDimScale(scale);
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void CheckerboardMaterial::setColorScale(PREC const &scale)
{
    _obj->setColorScale(scale);
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
